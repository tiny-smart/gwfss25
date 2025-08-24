import torch
import torch.distributed as dist


class ThreshController:
    ''' from CorrMatch: https://arxiv.org/abs/2306.04300 '''
    def __init__(self, momentum, thresh_init=0.7):

        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.gpu_num = dist.get_world_size()

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape
        pred_gather = torch.zeros([n * self.gpu_num, c, h, w]).cuda()
        # dist.all_gather_into_tensor(pred_gather, pred)
        dist._all_gather_base(pred_gather, pred)  # for low torch version

        pred = pred_gather
        if ignore_mask is not None:
            ignore_mask_gather = torch.zeros([n * self.gpu_num, h, w]).cuda().long()
            # dist.all_gather_into_tensor(ignore_mask_gather, ignore_mask)
            dist._all_gather_base(ignore_mask_gather, ignore_mask)  # for low torch version
            ignore_mask = ignore_mask_gather
        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        unique_cls = torch.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0
        for cls in unique_cls:
            cls_map = (mask_pred == cls)
            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        if cls_num > 0:
            return_dict['new_global'] = new_global / cls_num
        else:
            return_dict['new_global'] = None

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g and thresh['new_global'] is not None:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global
