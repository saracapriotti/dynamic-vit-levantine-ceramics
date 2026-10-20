import os
import sys

import timm
import torch
import torch.nn as nn

import config

# ── DynamicViT repo import (only needed for the LV-ViT static baseline) ────────
_DYNVIT_ROOT = os.environ.get("DYNAMICVIT_REPO", config.DYNAMICVIT_REPO)
if _DYNVIT_ROOT not in sys.path:
    sys.path.insert(0, _DYNVIT_ROOT)


def build_swin_small(num_classes: int, device: torch.device | None = None) -> nn.Module:
    """Swin-Small (ImageNet-pretrained, ~48.9M parameters)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"Swin-Small ready on {device}")
    return model


class LVViTClassifier(nn.Module):
    """Wraps LVViT_Teacher (plain, non-pruned LV-ViT-M) and returns only the cls logits,
    so it exposes the same forward signature as the other baselines."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x_cls, _ = self.backbone(x)
        return x_cls


def build_lvvit(
    num_classes: int,
    pretrained_path: str = config.PRETRAINED_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    """Static LV-ViT-M baseline: same backbone/weights as the DynamicViT-LVViT-M/0.7
    model, but with token pruning never applied (no score_predictor, no token dropping).
    Isolates the effect of dynamic pruning against an architecturally identical model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.dylvvit import LVViT_Teacher  # DynamicViT/models/dylvvit.py

    backbone = LVViT_Teacher(
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        p_emb=config.P_EMB,
        skip_lam=config.SKIP_LAM,
        return_dense=True,  # forward() always evaluates aux_head, must stay True
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    backbone.load_state_dict(state_dict, strict=False)  # drops the score_predictor.* keys

    backbone.head     = nn.Linear(backbone.head.in_features,     num_classes)
    backbone.aux_head = nn.Linear(backbone.aux_head.in_features, num_classes)

    model = LVViTClassifier(backbone).to(device)
    print(f"LV-ViT-M (static baseline) ready on {device}")
    return model


BUILDERS = {
    "swin_small": build_swin_small,
    "lvvit": build_lvvit,
}
