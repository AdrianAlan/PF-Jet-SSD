import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import tqdm
import yaml
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import trange

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.cuda.amp import GradScaler, autocast
from ssd.checkpoints import EarlyStopping
from ssd.layers.modules import MultiBoxLoss
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from ssd.qutils import get_delta, get_alpha, to_ternary
from utils import IsValidFile, Plotting, get_data_loader, set_logging


def get_loss_info(x):
    return ('Total loss {:.5f}, Localization {:.5f} ' +
            'Classification {:.5f} Regresion {:.5f}').format(x.sum(), *x)


def execute(rank, world_size, name, quantized, dataset, output, training_pref,
            ssd_settings, trained_model_path, verbose):

    setup(rank, world_size)

    if rank == 0:
        logname = '{}/{}.log'.format(output['model'], name)
        logger = set_logging('Train_SSD', logname, verbose)

    qbits = 8 if quantized else None
    ssd_settings['n_classes'] += 1
    plot = Plotting(save_dir=output['plots'])

    # Initialize dataset
    train_loader = get_data_loader(dataset['train'][rank],
                                   training_pref['batch_size_train'],
                                   training_pref['workers'],
                                   ssd_settings['input_dimensions'],
                                   ssd_settings['object_size'],
                                   rank,
                                   shuffle=True,
                                   return_pt=True,
                                   qbits=qbits)

    val_loader = get_data_loader(dataset['validation'][rank],
                                 training_pref['batch_size_validation'],
                                 training_pref['workers'],
                                 ssd_settings['input_dimensions'],
                                 ssd_settings['object_size'],
                                 rank,
                                 shuffle=False,
                                 return_pt=True,
                                 qbits=qbits)

    # Build SSD network
    ssd_net = build_ssd(rank, ssd_settings).to(rank)
    if rank == 0:
        logger.debug('SSD architecture:\n{}'.format(str(ssd_net)))

    # Initialize weights
    if trained_model_path:
        ssd_net.load_weights(trained_model_path)
    else:
        ssd_net.resnet.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.cnf.apply(weights_init)
        ssd_net.reg.apply(weights_init)

    # Data parallelization
    cudnn.benchmark = True
    net = DDP(ssd_net, device_ids=[rank])

    # Set training objective parameters
    optimizer = optim.SGD(net.parameters(), lr=1e-3,
                          momentum=training_pref['momentum'],
                          weight_decay=training_pref['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[20, 30, 50, 60,
                                                           70, 80, 90],
                                               gamma=0.5)
    if rank == 0:
        cp_es = EarlyStopping(patience=training_pref['patience'],
                              save_path='%s/%s.pth' % (output['model'], name))
    criterion = MultiBoxLoss(rank, ssd_settings['n_classes'],
                             min_overlap=ssd_settings['overlap_threshold'])
    scaler = GradScaler()
    verobse = verbose and rank == 0
    train_loss, val_loss = torch.empty(3, 0), torch.empty(3, 0)

    for epoch in range(1, training_pref['max_epochs']+1):

        # Start model training
        if verbose:
            tr = trange(len(train_loader), file=sys.stdout)
        all_epoch_loss = torch.zeros(3)
        net.train()

        # Ternarize weights
        if quantized:
            for m in net.modules():
                if is_first_or_last(m):
                    delta = get_delta(m.weight.data)
                    m.weight.delta = delta
                    m.weight.alpha = get_alpha(m.weight.data, delta)

        for batch_index, (images, targets) in enumerate(train_loader):

            # Ternarize weights
            if quantized:
                for m in net.modules():
                    if is_first_or_last(m):
                        m.weight.org = m.weight.data.clone()
                        m.weight.data = to_ternary(m.weight.data,
                                                   m.weight.delta,
                                                   m.weight.alpha)

            with autocast():
                outputs = net(images)
                l, c, r = criterion(outputs, targets)
                loss = l + c + r
            scaler.scale(loss).backward()

            if quantized:
                for m in net.modules():
                    if is_first_or_last(m):
                        m.weight.data.copy_(m.weight.org)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if quantized:
                for m in net.modules():
                    if is_first_or_last(m):
                        m.weight.org.copy_(m.weight.data.clamp_(-1, 1))

            all_epoch_loss += torch.tensor([l.item(), c.item(), r.item()])
            av_epoch_loss = all_epoch_loss / (batch_index + 1)

            info = 'Epoch {}, {}'.format(epoch, get_loss_info(av_epoch_loss))
            if verbose:
                tr.set_description(info)
                tr.update(1)

        if rank == 0:
            logger.debug(info)
        train_loss = torch.cat((train_loss, av_epoch_loss.unsqueeze(1)), 1)
        if verbose:
            tr.close()

        # Start model validation
        if verbose:
            tr = trange(len(val_loader), file=sys.stdout)
        all_epoch_loss = torch.zeros(3)
        net.eval()

        with torch.no_grad():

            # Ternarize weights
            if quantized:
                for m in net.modules():
                    if is_first_or_last(m):
                        m.weight.org = m.weight.data.clone()
                        m.weight.data = to_ternary(m.weight.data)

            for batch_index, (images, targets) in enumerate(val_loader):
                outputs = net(images)
                l, c, r = criterion(outputs, targets)
                l, c, r = reduce_tensor(l.data, c.data, r.data)
                all_epoch_loss += torch.tensor([l.item(), c.item(), r.item()])
                av_epoch_loss = all_epoch_loss / (batch_index + 1)
                info = 'Validation, {}'.format(get_loss_info(av_epoch_loss))
                if verbose:
                    tr.set_description(info)
                    tr.update(1)

            if rank == 0:
                logger.debug(info)
            val_loss = torch.cat((val_loss, av_epoch_loss.unsqueeze(1)), 1)
            if verbose:
                tr.close()

            plot.draw_loss(train_loss.cpu().numpy(),
                           val_loss.cpu().numpy(),
                           quantized=quantized)

            if rank == 0 and cp_es(av_epoch_loss.sum(0), ssd_net):
                break

            dist.barrier()

            if quantized:
                for m in net.modules():
                    if is_first_or_last(m):
                        m.weight.org.copy_(m.weight.data)
        scheduler.step()
    cleanup()


def is_first_or_last(layer):
    return (isinstance(layer, nn.Conv2d)
            and layer.in_channels > 3
            and layer.out_channels > 4)


def reduce_tensor(loc, cls, reg):
    lo, cl, rg = loc.clone(), cls.clone(), reg.clone()
    dist.all_reduce(lo)
    dist.all_reduce(cl)
    dist.all_reduce(rg)
    lo /= int(os.environ['WORLD_SIZE'])
    cl /= int(os.environ['WORLD_SIZE'])
    rg /= int(os.environ['WORLD_SIZE'])
    return lo, cl, rg


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11223'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train Single Shot Jet Detection Model')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-m', '--pre-trained-model', action=IsValidFile,
                        default=None, dest='pre_trained_model_path', type=str,
                        help='Path to pre-trained model')
    parser.add_argument('-t', '--ternary', action='store_true',
                        help='Ternarize weights')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    world_size = torch.cuda.device_count()

    mp.spawn(execute,
             args=(world_size,
                   args.name,
                   args.ternary,
                   config['dataset'],
                   config['output'],
                   config['training_pref'],
                   config['ssd_settings'],
                   args.pre_trained_model_path,
                   args.verbose),
             nprocs=world_size,
             join=True)
