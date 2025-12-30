"""
Core utilities to apply MHA2MLA-style conversion to a HF model.

This module provides:
- building a minimal args namespace
- patching weights (q/k reorder + kv low-rank)
- monkey-patching attention forward for Llama (Transformers 4.46.x compatible)

All code is self-contained in llava_mha2mla directory, no dependency on MHA2MLA repo.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, asdict
from typing import Any, Tuple

import torch

from .patch_model_load import patch_model
from .patching_llama_compat import mha2mla_llama_compat


@dataclass
class MHA2MLAConfig:
    # partial RoPE
    partial_rope_version: str = "high"  # high/low/uniform/2-norm
    rope_dim_for_mla: int = 16  # per-head RoPE dims used in MLA
    uniform_start_point: int = 0
    qk_tensor_path: str | None = None

    # low rank kv
    svd_init_method: str = "joint"  # none/split/joint/only_key/only_value
    low_rank: int = 8  # low_rank per kv head (paper uses low_rank * n_kv_heads as mid dim)

    # misc
    is_baseline: bool = False
    is_gqa2mha2mla: bool = False
    is_mla_from_scratch: bool = False


def apply_mha2mla_to_hf_causal_lm(
    *,
    repo_root: str,
    model: torch.nn.Module,
    model_config: Any,
    mha2mla_cfg: MHA2MLAConfig,
) -> Tuple[torch.nn.Module, list[torch.Tensor], list[torch.Tensor]]:
    """
    Converts a *causal LM* (e.g., LlamaForCausalLM / MistralForCausalLM) to an MLA-structured model.

    Args:
        repo_root: Root directory of the repository (not used anymore, kept for compatibility)
        model: The language model to convert
        model_config: Model configuration
        mha2mla_cfg: MHA2MLA configuration

    Returns:
      (patched_model, q_idx, k_idx)
    """
    mha2mla_args = types.SimpleNamespace(**asdict(mha2mla_cfg))

    # Patch model weights and structure
    mla_model, q_idx, k_idx = patch_model(model, model_config, mha2mla_args)

    # Monkey-patch attention forward (paper provides LLaMA-style patch; in modern Transformers,
    # Mistral is RoPE+KV-cache compatible and shares the same "position_embeddings" convention).
    #
    # We use the LLaMA patch for both LLaMA and Mistral when possible.
    mha2mla_llama_compat(q_idx, k_idx)

    # Store config so re-loading can reconstruct the modified modules before loading weights.
    mla_model.config.mha2mla = asdict(mha2mla_cfg)
    return mla_model, q_idx, k_idx
