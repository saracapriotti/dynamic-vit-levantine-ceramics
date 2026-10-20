"""
XAI utilities for DynamicViT:
  - Token-pruning visualisation (per sample, per class)
  - Keep-probability heatmaps
  - Token survival curves per class
  - Exit-depth + token-influence overlays
  - Patch sensitivity & predictor-score diagnostics
"""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from tqdm import tqdm

import config


# ── Image utilities ────────────────────────────────────────────────────────────

def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(config.IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std  = torch.tensor(config.IMAGENET_STD,  device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def _to_np_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """CHW tensor → HWC uint8 numpy array."""
    img = denormalize(img_tensor.cpu()).permute(1, 2, 0).numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _tokens_from_image(img_np: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """HWC uint8 → (196, 16, 16, 3) token array."""
    n = img_np.shape[0] // patch_size  # 14
    return img_np.reshape(n, patch_size, n, patch_size, 3).swapaxes(1, 2).reshape(-1, patch_size, patch_size, 3)


def _recover_image(tokens: np.ndarray, patch_size: int = 16) -> np.ndarray:
    n = int(tokens.shape[0] ** 0.5)  # 14
    return tokens.reshape(n, n, patch_size, patch_size, 3).swapaxes(1, 2).reshape(n * patch_size, n * patch_size, 3)


def collect_keep_indices(decisions, b_idx: int = 0) -> list[np.ndarray]:
    """Map relative kept indices → absolute patch indices (0..195) per stage."""
    alive = np.arange(196)
    keep_indices = []
    for stage in decisions:
        rel = stage[0][b_idx].detach().cpu().numpy()
        alive = alive[rel]
        keep_indices.append(alive.copy())
    return keep_indices


def _gen_masked_tokens(tokens: np.ndarray, kept_abs_idx: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    tokens = tokens.copy().astype(np.float32)
    drop   = np.setdiff1d(np.arange(196), kept_abs_idx)
    tokens[drop] = (1 - alpha) * tokens[drop] + alpha * 255.0
    return tokens.astype(np.uint8)


# ── Per-sample pruning visualisation ──────────────────────────────────────────

def visualize_pruning_sample(model, loader, sample_idx: int = 0, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    imgs, _ = next(iter(loader))
    img = imgs[sample_idx: sample_idx + 1].to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=img.is_cuda):
        _, decisions = model(img)

    keep_indices = collect_keep_indices(decisions)
    img_np = _to_np_uint8(img[0].cpu())
    tokens = _tokens_from_image(img_np)

    stage_imgs = [_recover_image(_gen_masked_tokens(tokens, ki)) for ki in keep_indices]

    n = len(stage_imgs) + 1
    _, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    for i, simg in enumerate(stage_imgs, 1):
        axes[i].imshow(simg)
        axes[i].set_title(f"Stage {i} ({len(keep_indices[i-1])} patches)")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# ── Per-class pruning grid ─────────────────────────────────────────────────────

def visualize_pruning_per_class(
    model,
    test_dataset,
    class_to_indices: dict,
    num_classes: int = 20,
    second_image_classes: set | None = None,
    third_image_classes:  set | None = None,
    fourth_image_classes: set | None = None,
    fifth_image_classes:  set | None = None,
    sixth_image_classes:  set | None = None,
    save_path: str = "pruning_per_class.png",
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    second_image_classes = second_image_classes or set()
    third_image_classes  = third_image_classes  or set()
    fourth_image_classes = fourth_image_classes or set()
    fifth_image_classes  = fifth_image_classes  or set()
    sixth_image_classes  = sixth_image_classes  or set()

    stages        = len(model.pruning_loc)
    cols_main     = 2
    rows_per_col  = num_classes // cols_main
    class_names   = [f"Class {i+1}" for i in range(num_classes)]

    fig, axes = plt.subplots(
        rows_per_col,
        (stages + 1) * cols_main,
        figsize=(4 * (stages + 1) * cols_main, 4 * rows_per_col),
    )
    axes = np.atleast_2d(axes)

    for cls_idx in range(num_classes):
        row        = cls_idx % rows_per_col
        col_offset = (cls_idx // rows_per_col) * (stages + 1)

        filtered = [
            idx for idx in class_to_indices[cls_idx]
            if "XPL" in str(test_dataset.dataset.df.iloc[test_dataset.indices[idx], 0])
        ]
        if not filtered:
            print(f"No XPL sample for class {cls_idx + 1}")
            continue

        if   cls_idx in sixth_image_classes  and len(filtered) > 5: pick = 5
        elif cls_idx in fifth_image_classes  and len(filtered) > 4: pick = 4
        elif cls_idx in fourth_image_classes and len(filtered) > 3: pick = 3
        elif cls_idx in third_image_classes  and len(filtered) > 2: pick = 2
        elif cls_idx in second_image_classes and len(filtered) > 1: pick = 1
        else:                                                         pick = 0

        img, _ = test_dataset[filtered[pick]]
        img    = img.unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=img.is_cuda):
            _, decisions = model(img)

        keep_indices = collect_keep_indices(decisions)
        img_np  = _to_np_uint8(img[0].cpu())
        tokens  = _tokens_from_image(img_np)
        stage_images = [_recover_image(_gen_masked_tokens(tokens, ki)) for ki in keep_indices]

        ax = axes[row, col_offset]
        ax.imshow(img_np)
        ax.set_title(class_names[cls_idx], fontsize=34, pad=3)
        if row == 0:
            ax.set_xlabel("Original", fontsize=34)

        for s_id, simg in enumerate(stage_images):
            ax = axes[row, s_id + 1 + col_offset]
            ax.imshow(simg)
            if row == 0:
                ax.set_title(f"Stage {s_id+1}\n{len(keep_indices[s_id])} patches", fontsize=34)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Keep-probability heatmap ───────────────────────────────────────────────────

def plot_keep_probability(
    model,
    loader,
    device=None,
    save_path: str | None = None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    pruning_layers = model.pruning_loc
    num_stages     = len(pruning_layers)
    num_patches    = 14 * 14

    counts_global  = np.zeros((num_stages, num_patches), dtype=np.int64)
    num_images     = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Keep-prob analysis"):
            if imgs.size(0) == 0:
                continue
            imgs = imgs.to(device)
            _, decisions = model(imgs)
            B = imgs.size(0)
            num_images += B

            for b in range(B):
                alive = np.arange(num_patches)
                for stage in range(num_stages):
                    keep_rel = decisions[stage][0][b].detach().cpu().numpy()
                    alive    = alive[keep_rel]
                    counts_global[stage, alive] += 1

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, num_stages + 1, width_ratios=[1] * num_stages + [0.04], wspace=0.05)
    cmap = "inferno"

    for stage in range(num_stages):
        ax = fig.add_subplot(gs[0, stage])
        prob14 = (counts_global[stage] / num_images).reshape(14, 14)
        prob7  = prob14.reshape(7, 2, 7, 2).mean(axis=(1, 3))

        sns.heatmap(
            prob7, square=True, cbar=False, annot=True, fmt=".2f",
            annot_kws={"size": 10}, linewidths=0.1, linecolor="white",
            vmin=0, vmax=1, cmap=cmap, ax=ax,
        )
        avg = counts_global[stage].sum() / num_images
        ax.set_title(f"Keep Prob – Layer {pruning_layers[stage]}\nAvg kept: {avg:.1f}/196", size=16)
        ax.axis("off")

    cax = fig.add_subplot(gs[0, -1])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Keep probability", rotation=270, size=20, labelpad=30)
    cbar.ax.tick_params(labelsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Token survival per class ───────────────────────────────────────────────────

def plot_token_survival(
    model,
    loader,
    device=None,
    save_path: str | None = None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    pruning_layers  = model.pruning_loc
    num_patch_tokens = 196

    token_depths_by_class = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in loader:
            if imgs.size(0) == 0:
                continue
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)
            _, token_info = model(imgs)

            for b in range(B):
                cls = int(labels[b].item())
                alive      = torch.zeros(num_patch_tokens + 1, dtype=torch.bool)
                alive[1:]  = True
                exit_depth = torch.zeros(num_patch_tokens + 1, dtype=torch.int)

                for step, layer_entry in enumerate(token_info):
                    active = layer_entry[0][b]
                    if active is None or active.numel() == 0:
                        continue
                    active = torch.unique(active.long())
                    curr   = torch.zeros_like(alive)
                    curr[active] = True
                    exited = alive & (~curr)
                    exit_depth[exited] = pruning_layers[step]
                    alive = curr

                exit_depth[alive] = pruning_layers[-1]
                token_depths_by_class[cls].append(exit_depth[1:].cpu())

    layers     = [0] + pruning_layers
    num_layers = len(layers)

    survival_by_class = {}
    global_counts     = np.zeros(num_layers)
    total_images      = 0

    for cls, depths_list in token_depths_by_class.items():
        counts = np.zeros(num_layers)
        for depths in depths_list:
            d = depths.numpy()
            for i, l in enumerate(layers):
                n = len(d) if l == 0 else int(np.sum(d >= l))
                counts[i]        += n
                global_counts[i] += n
            total_images += 1
        counts /= len(depths_list)
        survival_by_class[cls] = counts

    global_counts /= (total_images * num_patch_tokens)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sorted_cls = sorted(survival_by_class)
    colors     = plt.cm.inferno_r(np.linspace(0.1, 0.9, len(sorted_cls)))

    for cls, color in zip(sorted_cls, colors):
        ax.plot(layers, survival_by_class[cls], marker="o", linewidth=1.2,
                alpha=0.6, color=color, label=f"Class {cls+1}")

    ax.set_xlabel("Transformer layer", size=14, labelpad=12)
    ax.set_ylabel("Avg surviving tokens", size=14, labelpad=12)
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(fontsize=7.5, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()


# ── Exit-depth + token-influence overlay ──────────────────────────────────────

def _compute_exit_depth(decisions, pruning_layers: list, b_idx: int = 0, n: int = 14) -> np.ndarray:
    num_tokens = n * n
    alive      = np.arange(num_tokens)
    exit_depth = np.zeros(num_tokens, dtype=np.int32)

    for stage, keep in enumerate(decisions):
        keep_rel   = keep[0][b_idx].detach().cpu().numpy()
        alive_next = alive[keep_rel]
        exited     = np.setdiff1d(alive, alive_next, assume_unique=True)
        exit_depth[exited] = pruning_layers[stage]
        alive = alive_next

    exit_depth[alive] = pruning_layers[-1]
    return exit_depth


def _prune_patch(img: torch.Tensor, patch_idx: int, patch_size: int = 16, n: int = 14) -> torch.Tensor:
    img = img.clone()
    r, c = divmod(patch_idx, n)
    img[:, :, r * patch_size:(r+1) * patch_size, c * patch_size:(c+1) * patch_size] = 0
    return img


def visualize_all_rows_selected(
    model,
    test_dataset,
    class_to_indices: dict,
    selected_image_index_per_class: dict,
    keyword: str = "XPL",
    save_path: str = "analysis_overlay.png",
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    n_tokens_side = 14
    n_patch_tokens = n_tokens_side * n_tokens_side

    num_rows = len(selected_image_index_per_class)
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
    axes = np.atleast_2d(axes)

    for row_idx, (cls_idx, image_choice) in enumerate(selected_image_index_per_class.items()):
        key = cls_idx - 1
        if key not in class_to_indices:
            print(f"Class {cls_idx} not found"); continue

        filtered = [
            idx for idx in class_to_indices[key]
            if keyword.lower() in str(
                test_dataset.dataset.df.iloc[test_dataset.indices[idx], 0]
            ).lower()
        ]
        if not filtered:
            print(f"No {keyword} image for class {cls_idx}"); continue

        img_idx = filtered[min(image_choice, len(filtered) - 1)]
        img, _  = test_dataset[img_idx]
        img     = img.unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=img.is_cuda):
            logits, decisions = model(img)

        img_np     = denormalize(img[0]).permute(1, 2, 0).cpu().numpy()
        exit_depth = _compute_exit_depth(decisions, model.pruning_loc)
        exit_grid  = exit_depth.reshape(n_tokens_side, n_tokens_side)
        exit_full  = resize(exit_grid.astype(np.float32), img_np.shape[:2],
                            order=0, mode="reflect", anti_aliasing=False)

        # token influence via patch occlusion
        probs       = torch.softmax(logits[0], dim=-1)
        target_cls  = probs.argmax().item()
        orig_prob   = probs[target_cls].item()
        delta       = np.zeros(n_patch_tokens, dtype=np.float32)

        for i in range(n_patch_tokens):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=img.is_cuda):
                out = model(_prune_patch(img, i))
                new_logits = out[0] if isinstance(out, tuple) else out
            new_p    = torch.softmax(new_logits[0], dim=-1)[target_cls].item()
            delta[i] = max(0.0, orig_prob - new_p)

        delta_map  = resize(delta.reshape(n_tokens_side, n_tokens_side),
                            img_np.shape[:2], order=5, mode="reflect", anti_aliasing=True)
        delta_map  = gaussian_filter(delta_map, sigma=8)
        dmax       = delta_map.max()
        delta_norm = delta_map / (dmax + 1e-8) if dmax > 0 else delta_map

        ax = axes[row_idx]

        ax[0].imshow(img_np)
        ax[0].set_title(f"Original (Class {cls_idx})", size=28, pad=10)
        ax[0].axis("off")

        ax[1].imshow(img_np)
        im = ax[1].imshow(exit_full, cmap="viridis", alpha=0.4)
        ax[1].set_title("Exit Depth overlay", size=28, pad=10)
        ax[1].axis("off")
        cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=18)

        cmap_inf = plt.colormaps["inferno"]
        colored  = cmap_inf(delta_norm)
        colored[..., -1] = np.clip(delta_norm * 1.5, 0.1, 0.8)
        ax[2].imshow(img_np)
        ax[2].imshow(colored)
        ax[2].set_title("Token Influence overlay", size=28, pad=10)
        ax[2].axis("off")
        cb2 = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap_inf),
            ax=ax[2], fraction=0.046, pad=0.04,
        )
        cb2.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ── Diagnostics ────────────────────────────────────────────────────────────────

def verify_patch_sensitivity(model, image_tensor: torch.Tensor, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        _, dec_orig = model(image_tensor.to(device))

    perturbed = image_tensor.clone()
    perturbed[:, :, 0:16, 0:16] = 0

    with torch.no_grad():
        _, dec_pert = model(perturbed.to(device))

    idx_orig = dec_orig[0][0][0].cpu().numpy()
    idx_pert = dec_pert[0][0][0].cpu().numpy()

    print(f"Patches kept (original):   {len(idx_orig)}")
    print(f"Patches kept (perturbed):  {len(idx_pert)}")
    print(f"Patch(0,0) in original?    {0 in idx_orig}")
    print(f"Patch(0,0) in perturbed?   {0 in idx_pert}")


def analyze_predictor_scores(model, loader, device=None, num_samples: int = 100):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    scores_p0, scores_rest = [], []

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i * loader.batch_size >= num_samples:
                break
            imgs = imgs.to(device)

            x          = model.patch_embed(imgs)
            x          = x.flatten(2).transpose(1, 2)
            cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
            x          = torch.cat((cls_tokens, x), dim=1) + model.pos_embed

            pred_score = model.score_predictor[0](
                x[:, 1:], torch.ones(x.shape[0], 196, 1, device=device)
            )
            keep_probs = torch.exp(pred_score[:, :, 0])

            scores_p0.extend(keep_probs[:, 0].cpu().numpy())
            scores_rest.extend(keep_probs[:, 1:].mean(dim=1).cpu().numpy())

    import numpy as np
    mean_p0   = np.mean(scores_p0)
    mean_rest = np.mean(scores_rest)
    print(f"Mean keep-prob patch(0,0):   {mean_p0:.4f}")
    print(f"Mean keep-prob other patches: {mean_rest:.4f}")
    if mean_p0 > 0.999:
        print("WARNING: Predictor has extreme bias toward patch(0,0).")


if __name__ == "__main__":
    import os
    from dataset import build_dataloaders
    from model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, dataset, _, _, test_ds = build_dataloaders()

    num_classes = len(dataset.labels.unique())
    model = load_model(config.BEST_MODEL_PATH, num_classes=num_classes, device=device)
    model.viz_mode = True
    model.eval()

    out_dir = os.path.join(config.REPO_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving images to {out_dir}")

    class_to_indices = defaultdict(list)
    for i, orig_idx in enumerate(test_ds.indices):
        label = test_ds.dataset.labels.iloc[orig_idx]
        class_to_indices[label - 1].append(i)

    print("1/5  pruning sample...")
    visualize_pruning_sample(model, test_loader, sample_idx=0, device=device)
    plt.savefig(os.path.join(out_dir, "pruning_sample.png"), dpi=200, bbox_inches="tight")
    plt.close("all")

    print("2/5  keep-probability heatmap...")
    plot_keep_probability(model, test_loader, device=device,
                          save_path=os.path.join(out_dir, "keep_prob_heatmap.png"))
    plt.close("all")

    print("3/5  token survival curves...")
    plot_token_survival(model, test_loader, device=device,
                        save_path=os.path.join(out_dir, "token_survival.png"))
    plt.close("all")

    print("4/5  pruning per class grid...")
    visualize_pruning_per_class(
        model, test_ds, class_to_indices,
        save_path=os.path.join(out_dir, "pruning_per_class.png"),
        device=device,
    )
    plt.close("all")

    print("5/5  diagnostics...")
    verify_patch_sensitivity(model, next(iter(test_loader))[0][0:1], device=device)
    analyze_predictor_scores(model, test_loader, device=device)

    print(f"\nDone — images saved in {out_dir}/")
