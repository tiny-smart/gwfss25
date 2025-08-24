# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from .coco import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .coco_images import register_all_coco_unlabel
# from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .pascal_voc import load_voc_instances, register_pascal_voc
from .gwfss_semantic import (
    get_gwfss_semantic_files,
    load_gwfss_semantic,
)
from . import builtin as _builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if not k.startswith("_")]
