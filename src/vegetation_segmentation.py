"""Vegetation segmentation head on a frozen self-supervised backbone.

A lightweight approach to semantic segmentation when labeled data is scarce:
freeze a strong self-supervised backbone (DINOv2 ViT-S/14) and train only a
small convolutional head on top of its patch features. This needs far fewer
labels than training a full segmentation network and is cheap to run.

Classes (example, roadside vegetation): short mown grass, tall grass,
woody scrub, other vegetation.

This file is the architecture and the inference shape only; it ships no
trained weights. The backbone is loaded from torch hub at runtime.

Author: Moali Jaberi.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_W, IN_H = 896, 504                       # multiples of 14 for ViT/14 patches
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 4


def build_head(in_dim: int = 1536, hidden: int = 256,
               n_classes: int = NUM_CLASSES) -> nn.Sequential:
    """Small conv head over concatenated DINOv2 intermediate-layer features."""
    return nn.Sequential(
        nn.Conv2d(in_dim, hidden, 1), nn.GELU(),
        nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        nn.Conv2d(hidden, n_classes, 1),
    )


class VegetationSegmenter:
    """Frozen DINOv2 backbone + trained conv head."""

    def __init__(self, head_weights: str | None = None, device: str = "cpu"):
        self.device = device
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        self.head = build_head().to(device).eval()
        if head_weights:
            self.head.load_state_dict(
                torch.load(head_weights, map_location=device, weights_only=True))

    @torch.no_grad()
    def _features(self, rgb: np.ndarray) -> torch.Tensor:
        import cv2
        x = cv2.resize(rgb, (IN_W, IN_H)).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(self.device)
        feats = torch.cat(self.backbone.get_intermediate_layers(x, n=4), dim=-1)
        gh, gw = IN_H // 14, IN_W // 14
        return feats.reshape(gh, gw, -1).permute(2, 0, 1)[None].float()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, conf_threshold: float = 0.60):
        """Return (class_index_map, confidence_map) at the input resolution.

        Pixels below the confidence threshold are set to 255 (unassigned);
        the head has no background class, so callers gate it to vegetation
        regions (e.g. an excess-green test) before trusting a label.
        """
        h, w = rgb.shape[:2]
        logits = self.head(self._features(rgb))
        logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                               align_corners=False)
        prob = logits.softmax(1)[0]
        conf, pred = prob.max(0)
        conf = conf.cpu().numpy()
        pred = pred.cpu().numpy().astype(np.uint8)
        pred[conf < conf_threshold] = 255
        return pred, conf


# Training note (why this design):
#   - Backbone frozen: DINOv2 features already separate grass / scrub / trees
#     well, so only the head is trained. Order-of-magnitude fewer labels.
#   - Pre-train the head on a large open segmentation set, then fine-tune on a
#     small in-domain labeled set. In practice this cut the hardest confusion
#     (mown grass called tall grass) by roughly 4x versus in-domain only.
#   - Trees are handled by the object detector as trunks, not painted here.
