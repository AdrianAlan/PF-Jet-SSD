import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import match, log_sum_exp
from torch.autograd import Variable


class MultiBoxLoss(nn.Module):
    """Jet-SSD Loss Function
    This class produces confidence target indices by matching ground truth
    boxes with priors that have jaccard index grater than threshold parameter.
    Localization targets are produced by adding variance into offsets of ground
    truth boxes and their matched priors. Hard negative mining is added to
    filter the excessive number of negative examples that comes with using
    a large number of default bounding boxes. The Smooth L1 loss is implemented
    for the regression task.
    References:
    https://arxiv.org/pdf/1512.02325
    """

    def __init__(self,
                 rank,
                 priors,
                 n_classes,
                 min_overlap=0.5,
                 neg_pos=3):
        super(MultiBoxLoss, self).__init__()

        self.alpha = 0.1
        self.beta_loc = 1.0
        self.beta_cnf = 1.0
        self.beta_reg = 1.0
        self.device = torch.device(rank)
        self.defaults = priors.data
        self.n_classes = n_classes
        self.threshold = min_overlap
        self.negpos_ratio = neg_pos
        self.variance = .1

    def forward(self, predictions, targets):
        """Multibox loss calculation
        Args:
            predictions: a tuple containing loc preds, cnf preds, reg_preds
                cnf shape: [batch_size, num_priors, n_classes]
                loc shape: [batch_size, num_priors, 4]
                reg shape: [batch_size, num_priors, 1]
            targets: ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 6].
        Outputs:
            loss_l: localization loss
            loss_c: classification loss
            loss_r: regression loss
        """
        loc_data, cnf_data, reg_data = predictions

        bs = loc_data.size(0)  # batch size
        n_priors = self.defaults.size(0)  # number of priors

        # Match priors with ground truth boxes
        loc_truth = torch.Tensor(bs, n_priors, 2).to(self.device)
        cnf_truth = torch.LongTensor(bs, n_priors).to(self.device)
        reg_truth = torch.Tensor(bs, n_priors, 1).to(self.device)

        for i in range(bs):
            # Target
            target_coords = targets[i][:, :4].data  # Target coordinates
            target_labels = targets[i][:, 4].data  # Target labels
            target_regres = targets[i][:, -1:].data  # Target regression
            match(self.threshold,
                  target_coords,
                  self.defaults,
                  self.variance,
                  target_labels,
                  target_regres,
                  loc_truth,
                  cnf_truth,
                  reg_truth,
                  i)

        loc_truth = Variable(loc_truth, requires_grad=False)
        cnf_truth = Variable(cnf_truth, requires_grad=False)
        reg_truth = Variable(reg_truth, requires_grad=False)

        # Mask confidence
        pos = cnf_truth > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_truth)
        loc_prediction = loc_data[pos_idx].view(-1, 2)
        loc_truth = loc_truth[pos_idx].view(-1, 2)
        loss_l = F.smooth_l1_loss(loc_prediction, loc_truth, reduction='sum')

        # Compute max conf across batch for hard negative mining
        b_conf = cnf_data.view(-1, self.n_classes)
        loss_c = log_sum_exp(b_conf) - b_conf.gather(1, cnf_truth.view(-1, 1))
        loss_c = loss_c.view(bs, -1)
        loss_c[pos] = 0  # filter out pos boxes
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(cnf_data)
        neg_idx = neg.unsqueeze(2).expand_as(cnf_data)
        cnf_idx = (pos_idx+neg_idx).gt(0)
        trt_idx = (pos+neg).gt(0)
        cnf_prediction = cnf_data[cnf_idx].view(-1, self.n_classes)
        one_hot = F.one_hot(cnf_truth[trt_idx], 4)
        smooth_target = one_hot*(1-self.alpha) + self.alpha/4
        loss_c = F.binary_cross_entropy_with_logits(cnf_prediction,
                                                    smooth_target,
                                                    reduction='sum')

        # Compute regression loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(reg_data)
        reg_prediction = reg_data[pos_idx].view(-1, 1)
        reg_truth = reg_truth[pos_idx].view(-1, 1)
        loss_r = F.smooth_l1_loss(reg_prediction, reg_truth, reduction='sum')

        # Final normalized losses
        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_r /= N
        return self.beta_loc*loss_l, self.beta_cnf*loss_c, self.beta_reg*loss_r
