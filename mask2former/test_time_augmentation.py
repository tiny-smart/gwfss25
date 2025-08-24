# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import copy
import logging
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms import HFlipTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA

__all__ = [
    "SemanticSegmentorWithTTA",
]

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results
    
    def slide_inference(self, img, orig_shape, crop_size=(512,512), stride=(341,341), num_classes=4):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size = len(img)
        _, h_img, w_img = img[0]['image'].size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros((batch_size, num_classes, h_img, w_img)).cuda()
        count_mat = torch.zeros((batch_size, 1, h_img, w_img)).cuda()
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = copy.deepcopy(img)
                crop_img[0]['image'] = crop_img[0]['image'][:, y1:y2, x1:x2]
                crop_seg_logit = self.model(crop_img)[0]['sem_seg']
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if preds.shape[-2:] != orig_shape:
            preds = resize(
                preds,
                size=orig_shape,
                mode='bilinear',
                align_corners=False,
                warning=False)

        return [{'sem_seg': preds.squeeze()}]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        # final_predictions = self.model([input])[0].pop("sem_seg").flip(dims=[2])
                        final_predictions = self.slide_inference([input], orig_shape=orig_shape)[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions = self.slide_inference([input], orig_shape=orig_shape)[0].pop("sem_seg")
                else:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions += self.slide_inference([input], orig_shape=orig_shape)[0].pop("sem_seg").flip(dims=[2])
                    else:
                        final_predictions += self.slide_inference([input], orig_shape=orig_shape)[0].pop("sem_seg")

        final_predictions = final_predictions / count_predictions
        return {"sem_seg": final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms
