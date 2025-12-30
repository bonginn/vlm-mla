from __future__ import annotations

import json
import os
from typing import Any, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from huggingface_hub import snapshot_download

from .mha2mla_core import MHA2MLAConfig, apply_mha2mla_to_hf_causal_lm


def _is_dir(p: str) -> bool:
    return os.path.isdir(p)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_get_mha2mla_cfg_from_config(model_config: Any) -> MHA2MLAConfig | None:
    if not hasattr(model_config, "mha2mla"):
        return None
    d = model_config.mha2mla
    if isinstance(d, str):
        d = json.loads(d)
    if isinstance(d, dict):
        return MHA2MLAConfig(**d)
    return None


def load_llava_with_optional_mla(
    *,
    repo_root: str = "",  # Not used anymore, kept for compatibility
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    revision: str | None = None,
) -> Tuple[Any, Any]:
    """
    Load a LLaVA-like vision-language model from HF (or local dir).

    If `config.json` contains a `mha2mla` field, we will:
    - instantiate the original model from config
    - apply MHA2MLA conversion to the language_model (causal LM)
    - load weights from the directory

    Args:
        repo_root: Not used anymore, kept for compatibility
        model_name_or_path: Path to the model
        torch_dtype: Data type for the model
        device: Device to load the model on
        trust_remote_code: Whether to trust remote code
        local_files_only: Whether to use local files only
        revision: Model revision to load

    Returns:
      (model, processor_or_tokenizer)
    """

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=revision,
        )
    except Exception:
        processor = None

    # Try to load as Vision2Seq first (covers many LLaVA repos).
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        revision=revision,
    )
    mha2mla_cfg = _maybe_get_mha2mla_cfg_from_config(config)

    if mha2mla_cfg is None:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device)
        return model, (processor if processor is not None else AutoTokenizer.from_pretrained(model_name_or_path))

    # MLA: must rebuild modules before loading weights, because kv_proj/k_r_proj are new
    # and k_proj/v_proj are removed.
    base_model = AutoModelForVision2Seq.from_config(
        config, trust_remote_code=trust_remote_code
    )
    base_model.to(device=device, dtype=torch_dtype)

    # Most LLaVA implementations keep a `language_model` attribute.
    if not hasattr(base_model, "language_model"):
        raise ValueError(
            "This model does not expose `language_model`; cannot apply MHA2MLA safely."
        )
    lm = base_model.language_model
    lm_cfg = lm.config if hasattr(lm, "config") else AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    patched_lm, _, _ = apply_mha2mla_to_hf_causal_lm(
        repo_root="",  # Not used anymore, kept for compatibility
        model=lm,
        model_config=lm_cfg,
        mha2mla_cfg=mha2mla_cfg,
    )
    base_model.language_model = patched_lm

    # Now load weights from the MLA checkpoint itself into the patched module structure.
    # We cannot rely on `AutoModelForVision2Seq.from_pretrained()` here because it would
    # instantiate the *original* architecture (with k_proj/v_proj) and drop kv_proj/k_r_proj.
    # If model_name_or_path is not a local directory, download it from Hugging Face first.
    if not _is_dir(model_name_or_path):
        # Download from Hugging Face to a local cache directory
        print(f"Downloading model from Hugging Face: {model_name_or_path}")
        model_name_or_path = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Model downloaded to: {model_name_or_path}")

    from transformers.modeling_utils import load_sharded_checkpoint

    single_safe = os.path.join(model_name_or_path, "model.safetensors")
    shard_index_safe = os.path.join(model_name_or_path, "model.safetensors.index.json")
    single_bin = os.path.join(model_name_or_path, "pytorch_model.bin")
    shard_index_bin = os.path.join(model_name_or_path, "pytorch_model.bin.index.json")

    if os.path.exists(shard_index_safe) or os.path.exists(shard_index_bin):
        load_sharded_checkpoint(base_model, model_name_or_path)
    elif os.path.exists(single_safe):
        from safetensors.torch import load_file

        state_dict = load_file(single_safe)
        base_model.load_state_dict(state_dict, strict=True)
    elif os.path.exists(single_bin):
        state_dict = torch.load(single_bin, map_location="cpu")
        base_model.load_state_dict(state_dict, strict=True)
    else:
        # Fall back to HF helper (will error with a clear message)
        load_sharded_checkpoint(base_model, model_name_or_path)
    base_model.to(device=device, dtype=torch_dtype)

    return base_model, (processor if processor is not None else AutoTokenizer.from_pretrained(model_name_or_path))


