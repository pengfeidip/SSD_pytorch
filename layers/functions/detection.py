import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.keep_top_k = keep_top_k # keep top_k for per image

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0) #  number of default bboxes
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)

            #For each class, perform nms
            conf_scores = conf_preds[i].clone()

            # --- pengfei --- a mask tensor of non_background bounding boxes
            non_bg_mask = 1- (torch.argmax(conf_scores, dim=1) == 0)

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh) # 找到大于class_confidence_threshold 的bboxes
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)

        #我知道作者为什么要这样做了，
        # 因为作者为了更方便的统计各个类别的AP，因此分类保存检出结果, 这样导致每个必须每各类的tensor容器足够大
        flt = output.contiguous().view(num, -1, 5)

        # sort every bounding box by confidence
        _, idx = flt[:, :, 0].sort(1, descending=True)
        batch_idx = torch.tensor(range(num)).unsqueeze(1) # generate the batch idx

        flt = flt[batch_idx, idx, :]

        #  keep top k bounding boxes for every image
        #  (by set the rank greater than keep_top_k to 0)
        flt[:, self.keep_top_k:, :] = 0

        # already keep the keep_top_k, then return the origin
        _, rank = idx.sort(1)
        flt = flt[batch_idx, rank, :]
        new_output = flt.reshape(num, self.num_classes, self.top_k, 5)
        """
        --- pengfei --- 2019-1-9 11:23:34
        following is the original code, 
        I change it , because I want introduce the keep_top_k parameter
        """
        # flt = output.contiguous().view(num, -1, 5)
        # _, idx = flt[:, :, 0].sort(1, descending=True)
        # _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return new_output
