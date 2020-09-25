import torch

from torch.autograd import Function
from ..box_utils import decode, nms


class Detect(Function):
    """At inference Detect is the final layer of SSD.
    1) Decode location predictions.
    2) Apply non-maximum suppression to location predictions.
    3) Threshold to a top_k number of output predictions.
    """

    @staticmethod
    def forward(ctx, loc_data, conf_data, regr_data, prior_data,
                num_classes, top_k, conf_thresh, nms_thresh):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        variance = .1
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(batch_size, num_classes, top_k, 6)
        conf_preds = conf_data.view(batch_size,
                                    num_priors,
                                    num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, variance)

            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            regr_scores = regr_data[i].clone()
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                r_mask = c_mask.unsqueeze(1).expand_as(regr_scores)
                regres = regr_scores[r_mask].view(-1, 1)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]],
                               regres[ids[:count]]), 1)

        return output
