# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from utils import misc


def resolve_checkpoint_path(path):
    """Resolve a checkpoint path, downloading from HuggingFace if needed.

    Supports:
    - Local file paths (returned as-is)
    - HuggingFace: "hf://org/repo/filename.pt" -> auto-downloads and caches
    - manifold:// paths (returned as-is, internal only)
    """
    if path.startswith("hf://"):
        parts = path[len("hf://") :].split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2] if len(parts) > 2 else "model.pt"
        return hf_hub_download(repo_id, filename=filename)
    return path


def _load_model_from_checkpoint(checkpoint, model, strict):
    """Load model weights from a checkpoint dict."""
    model_dict = checkpoint["model"]

    # Build a filtered state dict that only contains keys matching the
    # current model in both name and shape. This allows loading checkpoints
    # that were saved with a different final layer patch size or other
    # architectural tweaks while keeping compatible weights.
    if hasattr(model, "module"):
        target_state = model.module.state_dict()
    else:
        target_state = model.state_dict()

    filtered_dict = {}
    skipped_keys = []
    for k, v in model_dict.items():
        if k in target_state:
            if v.shape == target_state[k].shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append((k, v.shape, target_state[k].shape))
        else:
            skipped_keys.append((k, v.shape, None))

    if misc.is_main_process():
        if skipped_keys:
            print("Warning: the following checkpoint keys were skipped due to shape/name mismatch:")
            for k, src_shape, tgt_shape in skipped_keys:
                print(f"  {k}: checkpoint shape={src_shape}  current shape={tgt_shape}")

    # Load only the filtered keys (non-strict) so missing keys in the
    # checkpoint are left as initialized in the model.
    if hasattr(model, "module"):
        model.module.load_state_dict(filtered_dict, strict=False)
    else:
        model.load_state_dict(filtered_dict, strict=False)


def save_checkpoint(cfg, model, optimizer, scheduler, iter_idx, only_latest=False):
    """
    Save a checkpoint of the training state.

    Args:
        cfg: Configuration object
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        iter_idx: Current iteration index
    """
    checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
    misc.makedirs(checkpoint_dir, exist_ok=True)

    # Get model state dict (handle DistributedDataParallel wrapper)
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iter_idx": iter_idx,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if not only_latest:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iter_idx:07d}.pt")
        misc.save_on_master(iter_idx, checkpoint, checkpoint_path)
        if misc.is_main_process():
            print(f"Saved checkpoint at iteration {iter_idx} to {checkpoint_path}")

    # Save a "latest" checkpoint for easy resuming
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    misc.save_on_master(iter_idx, checkpoint, latest_path)
    if misc.is_main_process():
        print(f"Saved latest checkpoint at iteration {iter_idx} to {latest_path}")


def load_checkpoint(
    cfg,
    model,
    optimizer,
    scheduler,
    test_only=False,
    # strict=True,
    strict=False,  # allow loading weights from checkpoints with different architectures (e.g., for ablation experiments)
):
    """
    Load a checkpoint if it exists.

    Priority order:
    1. cfg.checkpoint_path (explicit path, supports hf:// for HuggingFace)
    2. Local checkpoint_latest.pt in log_dir (for preemption recovery)
    3. Start from scratch

    Supports both full checkpoints (with optimizer/scheduler/iter_idx) and
    weights-only checkpoints (just {"model": state_dict}).

    Args:
        cfg: Configuration object
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into

    Returns:
        start_iter: Iteration to start from
    """
    # Default starting iteration
    start_iter = 0

    # Priority 1: Explicit checkpoint path (e.g., from release configs with hf:// paths)
    explicit_path = cfg.get("checkpoint_path", None)
    if explicit_path is not None:
        resolved_path = resolve_checkpoint_path(explicit_path)
        if misc.is_main_process():
            print(f"Loading checkpoint from {explicit_path}")

        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
        _load_model_from_checkpoint(checkpoint, model, strict)
        start_iter = checkpoint.get("iter_idx", -1) + 1

        if misc.is_main_process():
            print(f"Successfully loaded checkpoint from {explicit_path}")

    else:
        checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

        if os.path.exists(latest_path):
            # Priority 2: Load local checkpoint (preemption recovery)
            checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
            _load_model_from_checkpoint(checkpoint, model, strict)

            if not test_only:
                if "optimizer" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint["scheduler"])

            start_iter = checkpoint.get("iter_idx", -1) + 1

            if misc.is_main_process():
                print(
                    f"Resuming from checkpoint loaded from {latest_path} at iteration {start_iter}"
                )
        else:
            if misc.is_main_process():
                print("No checkpoint found, starting from scratch")

    # Make sure all processes are in sync
    if cfg.distributed:
        torch.distributed.barrier()

    return start_iter
