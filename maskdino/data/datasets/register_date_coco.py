# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


_THING_CLASSES = ["shone", "basketball"]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATASETS = {
    name: os.path.abspath(
        os.path.expanduser(
            os.environ.get(env_name, os.path.join(_REPO_ROOT, "datasets", name))
        )
    )
    for name, env_name in (
        ("date1", "DSPE_DATE1_ROOT"),
        ("date2", "DSPE_DATE2_ROOT"),
    )
}

_SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test",
}

_SOURCE_TO_TARGET = {
    "stone": "shone",
    "shone": "shone",
    "basketball": "basketball",
}


def _sanitize_date_json(dataset_name, split_name, json_file):
    """Keep only shone/basketball classes and remap category ids to a fixed order."""
    if not os.path.isfile(json_file):
        return json_file

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "datasets",
        "registered_date_coco",
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

    source_id_to_target_name = {}
    for cat in data.get("categories", []):
        target_name = _SOURCE_TO_TARGET.get(cat.get("name"))
        if target_name in _THING_CLASSES:
            source_id_to_target_name[cat["id"]] = target_name

    target_name_to_new_id = {name: idx + 1 for idx, name in enumerate(_THING_CLASSES)}
    data["categories"] = [
        {"id": target_name_to_new_id[name], "name": name}
        for name in _THING_CLASSES
    ]

    annotations = []
    for ann in data.get("annotations", []):
        target_name = source_id_to_target_name.get(ann.get("category_id"))
        if target_name is None:
            continue
        ann = dict(ann)
        ann["category_id"] = target_name_to_new_id[target_name]
        annotations.append(ann)
    data["annotations"] = annotations

    with open(sanitized_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return sanitized_json


def register_all_date_coco():
    for dataset_name, dataset_root in _DATASETS.items():
        for split_name, split_dir in _SPLITS.items():
            name = f"{dataset_name}_{split_name}"
            image_root = os.path.join(dataset_root, split_dir)
            json_file = os.path.join(image_root, "_annotations.coco.json")
            json_file = _sanitize_date_json(dataset_name, split_name, json_file)

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


register_all_date_coco()
