# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from .. import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def get_gwfss_semantic_files(image_dir, gt_dir, json_info=None):
    files = []
    # scan through the directory
    image_list = os.listdir(image_dir)
    print(f"{len(image_list)} images found in '{image_dir}'.")
    image_dict = {}
    for basename in image_list:
        image_file = os.path.join(image_dir, basename)
        suffix = ".png"
        assert basename.endswith(suffix), basename
        basename = os.path.basename(basename)[: -len(suffix)]

        image_dict[basename] = image_file
    
    for ann in os.listdir(gt_dir):
        image_file = image_dict.get(ann.split(".")[0], None)
        label_file = os.path.join(gt_dir, ann)
        segments_info = None
        if image_file is not None:
            files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def load_gwfss_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. 
        gt_dir (str): path to the raw annotations.

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    files = get_gwfss_semantic_files(image_dir, gt_dir)
    ret = []
    for image_file, label_file, segments_info in files:
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0]
                ),
                "sem_seg_file_name": label_file,
                "height": 512,
                "width": 512,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


_RAW_GWFSS_SEMANTIC_SPLITS = {
    f"gwfss_sem_seg_train": (
        "gwfss/gwfss_competition_train/images",
        "gwfss/gwfss_competition_train/class_id",
        f"None",
    ),
}


GWFSS_CATEGORIES = [
    {"color": (0, 0, 0), "isthing": 0, "id": 0, "trainId": 0, "name": "background"},
    {"color": (50, 255, 132), "isthing": 0, "id": 1, "trainId": 1, "name": "head"},
    {"color": (50, 132, 255), "isthing": 0, "id": 2, "trainId": 2, "name": "stem"},
    {"color": (214, 255, 50), "isthing": 0, "id": 3, "trainId": 3, "name": "leaf"},
]



def register_all_gwfss_semantic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in GWFSS_CATEGORIES]
    thing_colors = [k["color"] for k in GWFSS_CATEGORIES]
    stuff_classes = [k["name"] for k in GWFSS_CATEGORIES]
    stuff_colors = [k["color"] for k in GWFSS_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in GWFSS_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_GWFSS_SEMANTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir: load_gwfss_semantic(x, y)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )

