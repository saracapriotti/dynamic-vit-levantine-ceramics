import sys
import os
import torch
import torch.nn as nn

import config

# ── DynamicViT repo imports ────────────────────────────────────────────────────
# Entrambe le classi stanno in DynamicViT/models/dylvvit.py.
_DYNVIT_ROOT = os.environ.get("DYNAMICVIT_REPO", config.DYNAMICVIT_REPO)
if _DYNVIT_ROOT not in sys.path:
    sys.path.insert(0, _DYNVIT_ROOT)

from models.dylvvit import LVViTDiffPruning, LVViT_Teacher  # DynamicViT/models/dylvvit.py


def build_model(
    num_classes: int,
    pretrained_path: str = config.PRETRAINED_PATH,
    device: torch.device | None = None,
    viz_mode: bool = True,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LVViTDiffPruning(
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        p_emb=config.P_EMB,
        skip_lam=config.SKIP_LAM,
        return_dense=config.RETURN_DENSE,
        mix_token=config.MIX_TOKEN,
        pruning_loc=config.PRUNING_LOC,
        token_ratio=config.TOKEN_RATIO,
        viz_mode=viz_mode,
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.head     = nn.Linear(model.head.in_features,     num_classes)
    model.aux_head = nn.Linear(model.aux_head.in_features, num_classes)

    model.to(device)
    print(f"Student model ready on {device}")
    return model


def build_teacher(
    num_classes: int,
    pretrained_path: str = config.PRETRAINED_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = LVViT_Teacher(
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        p_emb=config.P_EMB,
        skip_lam=config.SKIP_LAM,
        return_dense=True,
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model"], strict=False)

    teacher.head     = nn.Linear(teacher.head.in_features,     num_classes).to(device)
    teacher.aux_head = nn.Linear(teacher.aux_head.in_features, num_classes).to(device)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Teacher model ready on {device} (frozen)")
    return teacher


def load_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)

    # support both full checkpoint dicts and plain state-dicts
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model weights from {checkpoint_path}")
    return model
