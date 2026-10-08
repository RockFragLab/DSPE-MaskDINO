# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


_THING_CLASSES = ["basketball", "stone"]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATASETS = {
    name: os.path.abspath(
        os.path.expanduser(
            os.environ.get(env_name, os.path.join(_REPO_ROOT, "datasets", name))
        )
    )
    for name, env_name in (
        ("fild9", "DSPE_FILD9_ROOT"),
        ("cocodate", "DSPE_COCODATE_ROOT"),
    )
}

_SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


def _sanitize_roboflow_json(dataset_name, split_name, json_file):
    """Drop Roboflow's category id 0 placeholder so Detectron2 maps classes to 0..N-1."""
    if not os.path.isfile(json_file):
        return json_file

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "datasets",
        "registered_roboflow_coco",
    )
    os.makedirs(cache_dir, exist_ok=True)
    sanitized_json = os.path.join(cache_dir, f"{dataset_name}_{split_name}.json")

    if (
        os.path.isfile(sanitized_json)
        and os.path.getmtime(sanitized_json) >= os.path.getmtime(json_file)
    ):
        return sanitized_json

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_category_ids = {
        cat["id"]
        for cat in data.get("categories", [])
        if cat.get("id") != 0 and cat.get("name") in _THING_CLASSES
    }
    data["categories"] = [
        cat for cat in data.get("categories", []) if cat.get("id") in valid_category_ids
    ]
    data["annotations"] = [
        ann for ann in data.get("annotations", []) if ann.get("category_id") in valid_category_ids
    ]

    with open(sanitized_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return sanitized_json


def register_all_stone_coco():
    for dataset_name, dataset_root in _DATASETS.items():
        for split_name, split_dir in _SPLITS.items():
            name = f"{dataset_name}_{split_name}"
            image_root = os.path.join(dataset_root, split_dir)
            json_file = os.path.join(image_root, "_annotations.coco.json")
            json_file = _sanitize_roboflow_json(dataset_name, split_name, json_file)

            DatasetCatalog.register(
                name,
                lambda json_file=json_file, image_root=image_root, name=name: load_coco_json(
                    json_file, image_root, name
                ),
            )
            MetadataCatalog.get(name).set(
                evaluator_type="coco_instance_extra",
                thing_classes=_THING_CLASSES,
                json_file=json_file,
                image_root=image_root,
            )


register_all_stone_coco()
