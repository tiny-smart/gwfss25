# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import re
import json

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.nn.functional as F
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
# from config.config import get_cfg
from config.add_cfg import add_ssl_config
# from detectron2.data import MetadataCatalog, build_detection_train_loader
from data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    # DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from modules.defaults import DefaultTrainer

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import hooks

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    ImageDatasetMapper,
    MaskFormerImageDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from mask2former.utils.lr_scheduler import build_lr_scheduler

from detectron2.data.detection_utils import read_image
from detectron2.modeling import DatasetMapperTTA
from fvcore.transforms import HFlipTransform
import pandas as pd


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


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)

            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("gwfss_unlabel_train")
            
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True)

            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("gwfss_unlabel_train")
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.INPUT = cfg.INPUT
            cfg_gan.DATASETS.TRAIN = ("gwfss_unlabel_train")

            if cfg.SSL.TRAIN_SSL:
                cfg_gan.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
                # cfg.DATALOADER.REPEAT_THRESHOLD = 10.0

            cfg_gan.freeze()
            cfg.freeze()
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            mapper_unl = MaskFormerImageDatasetMapper(cfg_gan, True) #ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.DATASETS.TRAIN = ("coco_2017_unlabel_train",)
            if cfg.SSL.TRAIN_SSL:
                cfg_gan.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            cfg.defrost()
            cfg_gan = cfg.clone()
            cfg_gan.DATASETS.TRAIN = ("coco_2017_unlabel_train",)
            cfg_gan.freeze()
            cfg.freeze()
            mapper_unl = ImageDatasetMapper(cfg_gan, True)
            return build_detection_train_loader(cfg, mapper=mapper), build_detection_train_loader(cfg_gan, mapper=mapper_unl)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper), None

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def slide_inference(cls, model, img, orig_shape, crop_size=(512,512), stride=(341,341), num_classes=4):
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
                crop_seg_logit = model(crop_img)[0]['sem_seg']
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
            
        return preds.squeeze()
    
    @classmethod
    def inference(cls, cfg, img_folder, model, output_folder='./outputs'):
        assert os.path.exists(img_folder), f'please check the path of img_folder:{img_folder}'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'prediction'), exist_ok=True)

        model.eval()
        tta_mapper = DatasetMapperTTA(cfg)
        for i, img_name in enumerate(os.listdir(img_folder)):
            print(f'[{i}/{len(os.listdir(img_folder))}]: {img_name}')
            img_path = os.path.join(img_folder, img_name)
            image = read_image(img_path, 'RGB')
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

            orig_shape = image.shape[1:]
            inputs = {'image':image, 'height':image.shape[1], 'width':image.shape[2]}
            augmented_inputs = tta_mapper(inputs)
            tfms = [x.pop("transforms") for x in augmented_inputs]

            final_predictions = None
            count_predictions = 0
            for input, tfm in zip(augmented_inputs, tfms):
                count_predictions += 1
                with torch.no_grad():
                    if final_predictions is None:
                        if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                            final_predictions = cls.slide_inference(model, [input], orig_shape=orig_shape).flip(dims=[2])
                        else:
                            final_predictions = cls.slide_inference(model, [input], orig_shape=orig_shape)
                    else:
                        if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                            final_predictions += cls.slide_inference(model, [input], orig_shape=orig_shape).flip(dims=[2])
                        else:
                            final_predictions += cls.slide_inference(model, [input], orig_shape=orig_shape)

            final_predictions = final_predictions / count_predictions

            pred = torch.argmax(final_predictions, dim=0).cpu().numpy()
            df = pd.DataFrame(pred)
            df.to_csv(os.path.join(output_folder, 'prediction', img_name.split('.')[0]+'.csv'), index=False, header=False)
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ssl_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Update dataloader for SSL case.
    if cfg.SSL.PERCENTAGE != 100:
        cfg.DATASETS.TRAIN = (cfg.DATASETS.TRAIN[0]+f"_{cfg.SSL.PERCENTAGE}",)
    # if cfg.SSL.TRAIN_SSL:
    #     cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"  # unlabeled data have no mask
    #     cfg.DATALOADER.REPEAT_THRESHOLD = 10.0
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def load_ckpt(model, ckpt_path, eval_who='None'):
    assert eval_who in ['TEACHER', 'STUDENT', 'None'], f"Invalid eval_who: {eval_who}"
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    print(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']
    new_ckpt = {}
    for k,v in ckpt.items():
        if eval_who == 'TEACHER' and k.startswith('modelTeacher'):
            new_ckpt[k.replace('modelTeacher.', '')] = v
        elif eval_who == 'STUDENT' and k.startswith('modelStudent'):
            new_ckpt[k.replace('modelStudent.', '')] = v
        elif eval_who == 'None':
            new_ckpt[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(f"missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")
    return missing_keys, unexpected_keys
    

def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    load_ckpt(model, ckpt_path='teacher_only.pth')
    Trainer.inference(
        cfg, 
        model=model,
        img_folder='/data/GWFSS/gwfss_competition_val/images', 
        output_folder='./test_pred')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
