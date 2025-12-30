from __future__ import annotations

import argparse
import os
import shutil

import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from .mha2mla_core import MHA2MLAConfig, apply_mha2mla_to_hf_causal_lm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=str, help="HF repo id or local path")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--torch_dtype", default="bfloat16", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--revision", default=None, type=str)

    # MHA2MLA knobs
    p.add_argument("--partial_rope_version", default="high", type=str)
    p.add_argument("--rope_dim_for_mla", default=16, type=int)
    p.add_argument("--uniform_start_point", default=0, type=int)
    p.add_argument("--qk_tensor_path", default=None, type=str)
    p.add_argument("--svd_init_method", default="joint", type=str)
    p.add_argument("--low_rank", default=8, type=int)
    args = p.parse_args()

    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown torch_dtype: {args.torch_dtype}")

    # Load model
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
        revision=args.revision,
    ).to(args.device)

    if not hasattr(model, "language_model"):
        raise ValueError("Model does not expose `language_model`; cannot convert.")
    lm = model.language_model
    lm_cfg = lm.config if hasattr(lm, "config") else cfg

    mha2mla_cfg = MHA2MLAConfig(
        partial_rope_version=args.partial_rope_version,
        rope_dim_for_mla=args.rope_dim_for_mla,
        uniform_start_point=args.uniform_start_point,
        qk_tensor_path=args.qk_tensor_path,
        svd_init_method=args.svd_init_method,
        low_rank=args.low_rank,
    )
    patched_lm, _, _ = apply_mha2mla_to_hf_causal_lm(
        repo_root="",  # Not used anymore, kept for compatibility
        model=lm,
        model_config=lm_cfg,
        mha2mla_cfg=mha2mla_cfg,
    )
    model.language_model = patched_lm

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Save processor/tokenizer alongside for convenience
    try:
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        proc.save_pretrained(args.output_dir)
    except Exception:
        pass

    # Also copy non-weight assets if converting from local dir (optional)
    if os.path.isdir(args.model):
        for fname in ["preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
            src = os.path.join(args.model, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Store mha2mla config into config.json for reload-time reconstruction
    cfg2 = AutoConfig.from_pretrained(args.output_dir, trust_remote_code=True)
    cfg2.mha2mla = {
        "partial_rope_version": mha2mla_cfg.partial_rope_version,
        "rope_dim_for_mla": mha2mla_cfg.rope_dim_for_mla,
        "uniform_start_point": mha2mla_cfg.uniform_start_point,
        "qk_tensor_path": mha2mla_cfg.qk_tensor_path,
        "svd_init_method": mha2mla_cfg.svd_init_method,
        "low_rank": mha2mla_cfg.low_rank,
        "is_baseline": False,
        "is_gqa2mha2mla": False,
        "is_mla_from_scratch": False,
    }
    cfg2.save_pretrained(args.output_dir)

    print(f"[ok] saved MLA-converted model to {args.output_dir}")


if __name__ == "__main__":
    main()


