from __future__ import annotations

import argparse
import gc
import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

from .load import load_llava_with_optional_mla
from .mla_cache import MLACache

def get_memory_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)

def get_max_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_memory_stats():
    torch.cuda.reset_peak_memory_stats()

def _build_prompt(processor, prompt: str) -> str:
    return f"USER: <image>\n{prompt}\nAnswer the question using a single word or short phrase. ASSISTANT:"

@torch.no_grad()
def run_generation(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    use_mla_cache: bool,
    force_length: bool,
    batch_size: int = 1,
) -> Dict[str, Any]:
    
    # Load Images - repeat for batch
    image = Image.open(image_path).convert("RGB")
    images = [image] * batch_size
    final_prompt = _build_prompt(processor, prompt)
    prompts = [final_prompt] * batch_size
    
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # For batch, input_ids shape is [batch_size, seq_len]
    # We'll track the average input length per sample
    batch_size_actual = inputs["input_ids"].shape[0]
    input_seq_len = inputs["input_ids"].shape[1]
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Generation
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=(max_new_tokens if force_length else 0),
        do_sample=False,
        use_cache=True,
        **(
            {"past_key_values": MLACache()}
            if use_mla_cache and hasattr(model.config, "mha2mla")
            else {}
        ),
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # out shape: [batch_size, seq_len]
    output_seq_len = out.shape[-1]
    gen_len = output_seq_len - input_seq_len
    
    # For batch processing, gen_len represents the sequence length difference
    # Each sample in the batch generates the same number of tokens (due to padding)
    avg_gen_len_per_sample = gen_len
    
    return {
        "latency": end_time - start_time,
        "gen_len": gen_len,  # Total tokens generated across batch
        "avg_gen_len_per_sample": avg_gen_len_per_sample,
        "tokens_per_sec": gen_len / (end_time - start_time) if gen_len > 0 else 0
    }

def evaluate_model(args, model_path: str, is_mla: bool, torch_dtype, batch_size: int = 1) -> Dict[str, float]:
    print(f"\n[{'MLA' if is_mla else 'Baseline'}] Loading model: {model_path}...")
    
    # Clean
    torch.cuda.empty_cache()
    gc.collect()
    reset_memory_stats()
    start_mem = get_memory_mb()

    # Load Model
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, processor = load_llava_with_optional_mla(
        repo_root=repo_root,
        model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        device="cuda",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    
    # Model Weight (Current - Before)
    model_mem = get_memory_mb() - start_mem
    print(f"Model Weights Memory: {model_mem:.2f} MB")

    # Warmup
    print("Warming up...")
    run_generation(
        model, processor, args.image, args.prompt, 
        max_new_tokens=10, 
        use_mla_cache=is_mla, 
        force_length=False,
        batch_size=batch_size
    )
    torch.cuda.empty_cache()
    reset_memory_stats()
    
    base_mem_after_warmup = get_memory_mb()
    
    # Testing...
    metrics = {"peak_mem_overhead": [], "gen_len": [], "avg_gen_len": [], "tps": []}
    
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs}...", end="\r")
        reset_memory_stats()
        
        res = run_generation(
            model, processor, args.image, args.prompt, 
            args.max_new_tokens, 
            use_mla_cache=is_mla, 
            force_length=args.force_length,
            batch_size=batch_size
        )
        
        peak_mem = get_max_memory_mb()
        
        # Peak - Base = KV Cache + Activation Overhead
        overhead = peak_mem - base_mem_after_warmup
        
        metrics["peak_mem_overhead"].append(overhead)
        metrics["gen_len"].append(res["gen_len"])
        metrics["avg_gen_len"].append(res.get("avg_gen_len_per_sample", res["gen_len"]))
        metrics["tps"].append(res["tokens_per_sec"])
        
        # Clean
        torch.cuda.empty_cache()

    print(f"Run {args.num_runs}/{args.num_runs} Done.")

    # Clean
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_weight_mb": model_mem,
        "kv_overhead_mb": np.mean(metrics["peak_mem_overhead"]),
        "tokens_per_sec": np.mean(metrics["tps"]),
        "avg_gen_len": np.mean(metrics["avg_gen_len"]),
        "total_gen_len": np.mean(metrics["gen_len"]),  # Total across batch
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_model", required=True, type=str, help="Path to original LLaVA model")
    p.add_argument("--mla_model", required=True, type=str, help="Path to MLA converted model")
    p.add_argument("--image", required=True, type=str, help="Image file path")
    p.add_argument("--prompt", default="Describe this image in detail.", type=str)
    p.add_argument("--max_new_tokens", default=512, type=int)
    p.add_argument("--num_runs", default=5, type=int, help="Number of runs to average")
    p.add_argument("--dtype", default="bfloat16", type=str)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--no_force_length", action="store_false", dest="force_length", help="Disable forced generation length")
    p.add_argument("--batch_size", default=1, type=int, help="Batch size for generation (default: 1)")
    p.set_defaults(force_length=True)
    
    args = p.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print("="*60)
    print(f"Benchmarking Memory & Throughput")
    print(f"Target Gen Length: {args.max_new_tokens}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Force Length: {args.force_length}")
    print("="*60)

    # Evaluate Baseline
    b_res = evaluate_model(args, args.baseline_model, is_mla=False, torch_dtype=torch_dtype, batch_size=args.batch_size)
    
    # Evaluate MLA-converted model
    m_res = evaluate_model(args, args.mla_model, is_mla=True, torch_dtype=torch_dtype, batch_size=args.batch_size)

    # Output result
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'MLA':<15} | {'Diff / Savings':<15}")
    print("-" * 80)
    
    # 1. Model Weight (Static Memory)
    print(f"{'Model Weights (MB)':<25} | {b_res['model_weight_mb']:<15.1f} | {m_res['model_weight_mb']:<15.1f} | {m_res['model_weight_mb'] - b_res['model_weight_mb']:<15.1f}")
    
    # 2. KV Cache + Activation (Dynamic Memory)
    b_kv = b_res['kv_overhead_mb']
    m_kv = m_res['kv_overhead_mb']
    kv_saving_pct = (b_kv - m_kv) / b_kv * 100 if b_kv > 0 else 0
    print(f"{'KV+Act Overhead (MB)':<25} | {b_kv:<15.1f} | {m_kv:<15.1f} | -{kv_saving_pct:.1f}%")
    
    # 3. Peak Memory
    b_peak = b_res['model_weight_mb'] + b_kv
    m_peak = m_res['model_weight_mb'] + m_kv
    peak_saving_pct = (b_peak - m_peak) / b_peak * 100 if b_peak > 0 else 0
    print(f"{'Total Peak Mem (MB)':<25} | {b_peak:<15.1f} | {m_peak:<15.1f} | -{peak_saving_pct:.1f}%")

    # 4. Throughput
    print("-" * 80)
    print(f"{'Throughput (tok/s)':<25} | {b_res['tokens_per_sec']:<15.2f} | {m_res['tokens_per_sec']:<15.2f} | {m_res['tokens_per_sec'] / b_res['tokens_per_sec']:.2f}x")
    
    print("-" * 80)
    print(f"Avg Gen Length per Sample: {b_res['avg_gen_len']:.1f} vs {m_res['avg_gen_len']:.1f}")
    if args.batch_size > 1:
        print(f"Total Tokens Generated (batch={args.batch_size}): {b_res.get('total_gen_len', b_res['avg_gen_len']*args.batch_size):.1f} vs {m_res.get('total_gen_len', m_res['avg_gen_len']*args.batch_size):.1f}")
    print("="*80)

if __name__ == "__main__":
    main()