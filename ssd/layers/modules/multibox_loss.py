import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import match, log_sum_exp
from torch.autograd import Variable


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    This class produces confidence target indices by matching ground truth
    boxes with priors that have jaccard index > threshold parameter.
    Localization targets are produced by adding variance into offsets of ground
    truth boxes and their matched priors. Hard negative mining is added to
    filter the excessive number of negative examples that comes with using
    a large number of default bounding boxes. https://arxiv.org/pdf/1512.02325
    """

    def __init__(self,
                 rank,
                 priors,
                 n_classes,
                 min_overlap=0.5,
                 neg_pos=3):
        super(MultiBoxLoss, self).__init__()

        self.beta_loc = 1.0
        self.beta_cnf = 1.0
        self.beta_reg = 1.0
        self.rank = rank
        self.priors = priors
        self.n_classes = n_classes
        self.threshold = min_overlap
        self.negpos_ratio = neg_pos
        self.variance = .1

    def forward(self, predictions, targets):
        """Multibox loss calculation
        Args:
            predictions: a tuple containing loc preds, conf preds, regr_preds
                         and prior boxes from SSD net.
                conf shape:   [batch_size, num_priors, n_classes]
                loc shape:    [batch_size, num_priors, 4]
                regr shape:   [batch_size, num_priors, 1]
                priors shape: [num_priors, 4]
            targets (tensor): ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 6].
        Outputs:
            loss_l: localization loss
            loss_c: classification loss
            loss_r: regression loss
        """
        loc_data, conf_data, regr_data = predictions
        defaults = self.priors[:loc_data.size(1), :].data

        bs = loc_data.size(0)  # batch size
        n_priors = self.priors.size(0)  # number of priors

        # Match priors with ground truth boxes
        loc_t = torch.Tensor(bs, n_priors, 2)
        conf_t = torch.LongTensor(bs, n_priors)
        regr_t = torch.Tensor(bs, n_priors, 1)
        for idx in range(bs):
            coords = targets[idx][:, :4].data  # truth coordinates
            labels = targets[idx][:, 4].data  # truth labels
            regres = targets[idx][:, -1:].data  # truth auxiliary regression
            match(self.threshold, coords, defaults, self.variance, labels,
                  regres, loc_t, conf_t, regr_t, idx)
        if self.rank != 'cpu':
            loc_t = loc_t.cuda(self.rank)
            conf_t = conf_t.cuda(self.rank)
            regr_t = regr_t.cuda(self.rank)
        else:
            loc_t = loc_t.cpu()
            conf_t = conf_t.cpu()
            regr_t = regr_t.cpu()
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        regr_t = Variable(regr_t, requires_grad=False)

        # Mask confidence
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_t)
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_t = loc_t[pos_idx].view(-1, 2)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        b_conf = conf_data.view(-1, self.n_classes)
        loss_c = log_sum_exp(b_conf) - b_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(bs, -1)
        loss_c[pos] = 0  # filter out pos boxes
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.n_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Compute regression loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(regr_data)
        regr_p = regr_data[pos_idx].view(-1, 1)
        regr_t = regr_t[pos_idx].view(-1, 1)
        loss_r = F.l1_loss(regr_p, regr_t, reduction='sum')

        # Final normalized losses
        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_r /= N
        return self.beta_loc*loss_l, self.beta_cnf*loss_c, self.beta_reg*loss_r
