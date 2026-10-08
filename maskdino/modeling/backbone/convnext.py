import logging
import os
import re

import timm
import torch
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class D2ConvNeXt(Backbone):
    """
    Detectron2 backbone wrapper for timm ConvNeXt / ConvNeXtV2 models.

    It exposes hierarchical features as res2-res5 so the existing MaskDINO
    pixel decoder can consume it without structural changes.
    """

    _FEATURE_NAME_TO_INDEX = {
        "res2": 0,
        "res3": 1,
        "res4": 2,
        "res5": 3,
    }

    def __init__(self, cfg, input_shape):
        super().__init__()
        self._logger = logging.getLogger(__name__)

        model_name = cfg.MODEL.CONVNEXT.MODEL_NAME
        out_features = list(cfg.MODEL.CONVNEXT.OUT_FEATURES)
        unsupported = [name for name in out_features if name not in self._FEATURE_NAME_TO_INDEX]
        if unsupported:
            raise ValueError(f"Unsupported ConvNeXt output features: {unsupported}")

        out_indices = tuple(self._FEATURE_NAME_TO_INDEX[name] for name in out_features)
        pretrained = bool(cfg.MODEL.CONVNEXT.PRETRAINED and not cfg.MODEL.CONVNEXT.WEIGHTS)

        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained,
            in_chans=input_shape.channels,
            drop_path_rate=cfg.MODEL.CONVNEXT.DROP_PATH_RATE,
        )

        channels = self.backbone.feature_info.channels()
        strides = self.backbone.feature_info.reduction()

        self._out_features = out_features
        self._out_feature_channels = {
            name: channels[idx] for idx, name in enumerate(self._out_features)
        }
        self._out_feature_strides = {
            name: strides[idx] for idx, name in enumerate(self._out_features)
        }
        self._size_divisibility = max(self._out_feature_strides.values())

        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)
        self._load_backbone_weights(cfg.MODEL.CONVNEXT.WEIGHTS)

    def _freeze_backbone(self, freeze_at):
        if freeze_at <= 0:
            return

        freeze_prefixes = []
        if freeze_at >= 1:
            freeze_prefixes.extend(["stem_0", "stem_1"])
        for stage_idx in range(max(0, freeze_at - 1)):
            freeze_prefixes.append(f"stages_{stage_idx}")

        for name, parameter in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                parameter.requires_grad = False

    def _load_backbone_weights(self, weights_path):
        if not weights_path:
            return

        weights_path = os.path.expanduser(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"ConvNeXt weights not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported ConvNeXt checkpoint format: {weights_path}")

        model_state = self.backbone.state_dict()
        remapped_state = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue

            new_key = key.replace("module.", "")
            new_key = new_key.replace("backbone.", "")
            new_key = re.sub(r"^stem\.(\d+)\.", r"stem_\1.", new_key)
            new_key = re.sub(r"^stages\.(\d+)\.", r"stages_\1.", new_key)

            if new_key in model_state and model_state[new_key].shape == value.shape:
                remapped_state[new_key] = value

        incompatible = self.backbone.load_state_dict(remapped_state, strict=False)
        self._logger.info(
            "Loaded ConvNeXt weights from %s. matched=%d missing=%d unexpected=%d",
            weights_path,
            len(remapped_state),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    def forward(self, x):
        assert x.dim() == 4, f"ConvNeXt backbone expects NCHW input, got {x.shape}"
        features = self.backbone(x)
        return {name: features[idx] for idx, name in enumerate(self._out_features)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility
