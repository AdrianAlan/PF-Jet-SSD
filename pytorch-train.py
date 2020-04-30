import argparse
import h5py
import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import tqdm

from tqdm import trange
from torch.autograd import Variable

from ssd.checkpoints import EarlyStopping
from ssd.layers.modules import MultiBoxLoss
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from utils import Plotting


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def get_data_loader(source_path, batch_size, num_workers, shuffle=True):
    h5 = h5py.File(source_path, 'r')
    generator = CalorimeterJetDataset(hdf5_dataset=h5)
    return torch.utils.data.DataLoader(generator,
                                       batch_size=batch_size,
                                       collate_fn=detection_collate,
                                       shuffle=shuffle,
                                       num_workers=num_workers), h5


def batch_step(x, y, optimizer, net, criterion):
    x = Variable(x.cuda())
    y = [Variable(ann.cuda()) for ann in y]
    if optimizer:
        optimizer.zero_grad()
    output = net(x)
    return criterion(output, y)


def execute(model_name, qtype, train_dataset_path, val_dataset_path, save_dir,
            num_classes, num_workers, batch_size, train_epochs,
            overlap_threshold=0.5, es_patience=20, trained_model_path=None):

    num_classes += 1
    quantized = (qtype == 'binary') or (qtype == 'ternary')
    plot = Plotting(save_path='%s/%s-loss.png' % (save_dir, model_name))

    # Initialize dataset
    train_loader, h5t = get_data_loader(train_dataset_path, batch_size,
                                        num_workers)
    val_loader, h5v = get_data_loader(val_dataset_path, batch_size,
                                      num_workers, shuffle=False)

    # Build SSD network
    ssd_net = build_ssd('train', num_classes, qtype=qtype)
    print(ssd_net)

    # Initialize weights
    if trained_model_path:
        ssd_net.load_weights(trained_model_path)
    else:
        ssd_net.vgg.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # Data parallelization
    cudnn.benchmark = True
    net = torch.nn.DataParallel(ssd_net)
    net = net.cuda()

    # Print total number of parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total network parameters: %s' % total_params)

    # Set training objective parameters
    if quantized:
        lr = 1e-3
        momentum = 0.9
        weight_decay = 5e-4
        lrs = [5e-4, 1e-4, 5e-5, 1e-5]
        lr_steps = [20, 30, 40, 45]
    else:
        lr = 1e-3
        momentum = 0.9
        weight_decay = 5e-4
        lrs = [5e-4, 1e-4, 5e-5, 1e-5]
        lr_steps = [20, 30, 40, 45]
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    cp_es = EarlyStopping(patience=es_patience,
                          save_path='%s/%s.pth' % (save_dir, model_name))
    criterion = MultiBoxLoss(num_classes, overlap_threshold, 3, True)

    # Training
    train_loss, train_loss_l, train_loss_c = np.empty(0), np.empty(0), np.empty(0)
    val_loss, val_loss_l, val_loss_c = np.empty(0), np.empty(0), np.empty(0)

    for epoch in range(1, train_epochs+1):

        if epoch in lr_steps:
            new_lr = lrs[lr_steps.index(epoch)]
            adjust_learning_rate(optimizer, new_lr)

        # Start of model training
        tr = trange(len(train_loader)*batch_size, file=sys.stdout)
        tr.set_description('Epoch {}'.format(epoch))
        batch_loss_l, batch_loss_c = np.empty(0), np.empty(0)
        net.train()

        for batch_index, (images, targets) in enumerate(train_loader):

            loss_l, loss_c = batch_step(images, targets, optimizer,
                                        net, criterion)
            batch_loss_l = np.append(batch_loss_l, loss_l.item())
            batch_loss_c = np.append(batch_loss_c, loss_c.item())
            loss = loss_l + loss_c
            loss.backward()

            if quantized:
                for p in list(net.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)

            optimizer.step()

            if quantized:
                for p in list(net.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

            av_train_loss_l = np.average(batch_loss_l)
            av_train_loss_c = np.average(batch_loss_c)
            av_train_loss = av_train_loss_l + av_train_loss_c

            tr.set_description(
                ('Epoch {} Loss {:.5f} ' +
                 'Localization {:.5f} Classification {:.5f}').format(
                    epoch, av_train_loss, av_train_loss_l, av_train_loss_c))

            tr.update(len(images))

        train_loss_l = np.append(train_loss_l, av_train_loss_l)
        train_loss_c = np.append(train_loss_c, av_train_loss_c)
        train_loss = np.append(train_loss, av_train_loss)

        tr.close()

        # Validating

        tr = trange(len(val_loader)*batch_size, file=sys.stdout)
        tr.set_description('Validation')
        batch_loss_l, batch_loss_c = np.empty(0), np.empty(0)
        net.eval()

        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(val_loader):

                loss_l, loss_c = batch_step(images, targets, None,
                                            net, criterion)
                batch_loss_l = np.append(batch_loss_l, loss_l.item())
                batch_loss_c = np.append(batch_loss_c, loss_c.item())

                av_val_loss_l = np.average(batch_loss_l)
                av_val_loss_c = np.average(batch_loss_c)
                av_val_loss = av_val_loss_l + av_val_loss_c

                tr.set_description(
                    ('Validation Loss {:.5f} ' +
                     'Localization {:.5f} Classification {:.5f}').format(
                       av_val_loss, av_val_loss_l, av_val_loss_c))

                tr.update(len(images))

        tr.close()

        val_loss_l = np.append(val_loss_l, av_val_loss_l)
        val_loss_c = np.append(val_loss_c, av_val_loss_c)
        val_loss = np.append(val_loss, av_val_loss)

        plot.draw_loss([train_loss, train_loss_l, train_loss_c],
                       [val_loss, val_loss_l, val_loss_c],
                       ['Full', 'Localization', 'Classification'])

        if cp_es(av_val_loss, ssd_net):
            break

    h5t.close()
    h5v.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train Single Shot Jet Detection Model')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('qtype', type=str,
                        choices={'full', 'ternary', 'binary'},
                        help='Type of quantization')
    parser.add_argument('train_dataset', type=str,
                        help='Path to training dataset')
    parser.add_argument('validation_dataset', type=str,
                        help='Path to validation dataset')
    parser.add_argument('-s', '--save-dir', type=str, default='./models',
                        help='Path to save directory', dest='save_dir')
    parser.add_argument('-c', '--classes', type=int, default=1,
                        help='Number of target classes', dest='num_classes')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of workers', dest='num_workers')
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training epochs', dest='train_epochs')
    parser.add_argument('-t', '--overlap-threshold', type=float, default='0.5',
                        help='IoU threshold', dest='overlap_threshold')
    parser.add_argument('-p', '--patience', type=int, default='20',
                        help='Early stopping patince', dest='es_patience')
    parser.add_argument('-m', '--pre-trained', type=str,
                        default=None, help='Path to pre-trained model',
                        dest='trained_model_path')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    execute(args.name,
            args.qtype,
            args.train_dataset,
            args.validation_dataset,
            args.save_dir,
            args.num_classes,
            args.num_workers,
            args.batch_size,
            args.train_epochs,
            overlap_threshold=args.overlap_threshold,
            es_patience=args.es_patience,
            trained_model_path=args.trained_model_path)
