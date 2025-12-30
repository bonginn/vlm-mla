"""
Generate 2-norm rank tensor for LLaVA models.

This script computes the importance ranking of RoPE dimensions based on
the 2-norm of query-key interactions across calibration data.
"""
from __future__ import annotations

import argparse
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

# Import attention classes
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention


def cal_2_norm(states):
    """Calculate 2-norm for RoPE dimensions.
    
    Args:
        states: [bsz, q_len, num_heads, head_dim] where head_dim is structured as
                pairs of (real, imaginary) for each RoPE frequency.
    
    Returns:
        [bsz, q_len, num_heads, head_dim//2] - 2-norm for each frequency pair
    """
    # Match the original implementation exactly
    # Reshape: [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, num_heads, 2, head_dim//2]
    # Transpose: -> [bsz, q_len, num_heads, head_dim//2, 2]
    # Norm along dim=4 (last dim): -> [bsz, q_len, num_heads, head_dim//2]
    states = torch.norm(
        states.reshape(
            states.shape[0], states.shape[1], states.shape[2], 2, -1
        ).transpose(-1, -2),
        p=2,
        dim=4,
    )
    return states


def main():
    p = argparse.ArgumentParser(description="Generate 2-norm rank tensor for LLaVA models")
    p.add_argument("--model", required=True, type=str, help="Path to LLaVA model")
    p.add_argument("--output_path", required=True, type=str, help="Output path for rank tensor (.pth file)")
    p.add_argument("--dataset", default="liuhaotian/LLaVA-Instruct-150K", type=str, 
                   help="Dataset for calibration (default: LLaVA-Instruct-150K)")
    p.add_argument("--image_root", required=True, type=str, help="Root directory for images (e.g., data/coco2017/train2017)")
    p.add_argument("--sample_size", default=1024, type=int, help="Number of samples for calibration")
    p.add_argument("--batch_size", default=1, type=int, help="Batch size")
    p.add_argument("--max_length", default=2048, type=int, help="Max sequence length")
    p.add_argument("--device", default="cuda", type=str, help="Device to use")
    p.add_argument("--dtype", default="bfloat16", type=str, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()

    # Set dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Load model and processor
    print(f"Loading model from {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Get language model config
    if not hasattr(model, "language_model"):
        raise ValueError("Model does not expose `language_model`; cannot generate 2-norm for LLaVA.")
    lm = model.language_model
    lm_config = lm.config if hasattr(lm, "config") else config

    # Get model dimensions
    num_layers = lm_config.num_hidden_layers
    num_heads = lm_config.num_attention_heads
    num_kv_heads = getattr(lm_config, "num_key_value_heads", num_heads)
    head_dim = getattr(lm_config, "head_dim", lm_config.hidden_size // num_heads)
    group_size = num_heads // num_kv_heads

    print(f"Model config: {num_layers} layers, {num_heads} heads, {num_kv_heads} kv_heads, head_dim={head_dim}")

    # Register hooks to capture hidden states
    hidden_states_dict = {}

    def create_hook_fn(name):
        def hook(module, args, kwargs, output):
            # Capture hidden_states from kwargs (for newer transformers versions)
            if "hidden_states" in kwargs:
                hidden_states_dict[name] = kwargs["hidden_states"]
            # Also try to get from args (for older versions)
            elif len(args) > 0:
                hidden_states_dict[name] = args[0]
        return hook

    # Register hooks on attention modules
    attn_modules = {}
    for name, module in lm.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention)):
            hook_fn = create_hook_fn(name)
            module.register_forward_hook(hook_fn, with_kwargs=True)
            attn_modules[name] = module
            print(f"Registered hook for {name}")

    # Load calibration dataset with images
    print(f"Loading calibration dataset: {args.dataset}...")
    print(f"Image root: {args.image_root}")
    
    # Load dataset
    try:
        # Try streaming mode first for robustness
        ds_stream = load_dataset(args.dataset, split="train", streaming=True, trust_remote_code=True)
        ds_list = []
        for i, item in enumerate(ds_stream):
            ds_list.append(item)
            if len(ds_list) >= args.sample_size:
                break
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1} examples...")
        from datasets import Dataset
        train_dataset = Dataset.from_list(ds_list)
        print(f"Loaded {len(train_dataset)} examples from streaming dataset")
    except Exception as e:
        print(f"Streaming failed: {e}, trying regular load...")
        train_dataset = load_dataset(args.dataset, split="train", trust_remote_code=True)
        if len(train_dataset) > args.sample_size:
            train_dataset = train_dataset.select(range(args.sample_size))
    
    # Helper function to extract text from LLaVA conversations
    def _strip_image_tokens(s: str) -> str:
        return (
            s.replace("<image>\n", "")
            .replace("<image>", "")
            .replace("<ImageHere>", "")
            .strip()
        )
    
    def _pick_first_user_message(conv):
        """Extract the first user message from LLaVA conversation."""
        for msg in conv:
            if isinstance(msg, dict):
                from_role = (msg.get("from") or msg.get("role") or "").lower()
                if from_role in {"human", "user"}:
                    value = msg.get("value", "")
                    if isinstance(value, str):
                        return _strip_image_tokens(value)
                    elif isinstance(value, list):
                        # Extract text from content list
                        text_parts = []
                        for c in value:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text_parts.append(c.get("text", ""))
                            elif isinstance(c, str):
                                text_parts.append(c)
                        return _strip_image_tokens(" ".join(text_parts))
        return None
    
    # Prepare dataset items with images and text
    dataset_items = []
    for i, item in enumerate(train_dataset):
        if i >= args.sample_size:
            break
        
        # Get image path
        img_name = item.get("image", None)
        if img_name is None:
            continue
        
        img_path = os.path.join(args.image_root, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue
        
        # Extract text from conversations
        user_text = None
        if "conversations" in item:
            user_text = _pick_first_user_message(item["conversations"])
        elif "text" in item:
            user_text = item["text"]
        
        if not user_text:
            continue
        
        dataset_items.append({
            "image_path": img_path,
            "user_text": user_text,
        })
    
    print(f"Prepared {len(dataset_items)} items with images and text")
    
    if len(dataset_items) == 0:
        raise ValueError("No valid items found in dataset. Check image_root path and dataset format.")
    
    # Limit to sample_size
    if len(dataset_items) > args.sample_size:
        dataset_items = dataset_items[:args.sample_size]

    # Initialize accumulators
    query_states = [[] for _ in range(num_layers)]
    key_states = [[] for _ in range(num_layers)]

    print(f"Processing {len(dataset_items)} samples with images...")
    p_bar = tqdm(total=len(dataset_items))

    with torch.no_grad():
        # Process items one by one (batch_size=1 for simplicity with images)
        for item in dataset_items:
            try:
                # Load image
                image = Image.open(item["image_path"]).convert("RGB")
                
                # Prepare prompt (LLaVA format with image token)
                prompt = item["user_text"]
                if "<image>" not in prompt:
                    prompt = "<image>\n" + prompt
                
                # Process with processor (handles both image and text)
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                
                # Clear hidden states from previous iteration
                hidden_states_dict.clear()
                
                # Forward pass through full LLaVA model (including vision tower)
                # This ensures hidden states include visual information
                # We don't need the output, just the forward pass to trigger hooks
                with torch.no_grad():
                    _ = model(**inputs, output_attentions=False, output_hidden_states=False)
                
                p_bar.update(1)
                
                # Extract query and key states from each layer
                for name, module in attn_modules.items():
                    if name not in hidden_states_dict:
                        continue
                    
                    # Extract layer index from module name
                    # Format: "model.layers.0.self_attn" -> 0
                    parts = name.split(".")
                    layer_idx = None
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                break
                            except ValueError:
                                continue
                    
                    if layer_idx is None:
                        continue
                    
                    hidden_states = hidden_states_dict[name]
                    bsz, q_len, _ = hidden_states.shape
                    
                    # Compute Q and K projections (matching original exactly)
                    q = module.q_proj(hidden_states).reshape(
                        bsz, q_len, num_heads, head_dim
                    )  # [bsz, q_len, num_heads, head_dim]
                    k = module.k_proj(hidden_states).reshape(
                        bsz, q_len, num_kv_heads, head_dim
                    )  # [bsz, q_len, num_kv_heads, head_dim]
                    
                    # Calculate 2-norm (matching original: cal_2_norm then mean over seq_len)
                    query_states[layer_idx].append(
                        cal_2_norm(q).mean(dim=1, keepdim=False).cpu()
                    )  # [bsz, num_heads, head_dim//2]
                    key_states[layer_idx].append(
                        cal_2_norm(k).mean(dim=1, keepdim=False).cpu()
                    )  # [bsz, num_kv_heads, head_dim//2]
                    
            except Exception as e:
                print(f"Error processing item: {e}")
                import traceback
                traceback.print_exc()
                continue

    p_bar.close()

    # Aggregate across samples (matching original implementation exactly)
    print("Aggregating statistics...")
    # Check if we have data for all layers
    for i in range(num_layers):
        if len(query_states[i]) == 0:
            print(f"Warning: No data collected for layer {i}")
            # Create dummy data to avoid errors
            query_states[i] = [torch.zeros(1, num_heads, head_dim // 2)]
            key_states[i] = [torch.zeros(1, num_kv_heads, head_dim // 2)]
    
    query_states = torch.stack(
        [torch.cat(query_states[i], dim=0) for i in range(num_layers)], dim=0
    )  # [num_layers, sample_size, num_heads, head_dim//2]
    key_states = torch.stack(
        [torch.cat(key_states[i], dim=0) for i in range(num_layers)], dim=0
    )  # [num_layers, sample_size, num_kv_heads, head_dim//2]
    
    # Average over samples
    query_states = torch.mean(query_states, dim=1, keepdim=False)  # [num_layers, num_heads, head_dim//2]
    key_states = torch.mean(key_states, dim=1, keepdim=False)  # [num_layers, num_kv_heads, head_dim//2]
    
    # Expand key_states to match query heads if GQA (matching original)
    key_states = (
        key_states.unsqueeze(2)
        .expand(
            num_layers,
            num_kv_heads,
            group_size,
            head_dim // 2,
        )
        .reshape(num_layers, num_heads, head_dim // 2)
    )  # [num_layers, num_heads, head_dim//2]
    
    # Compute qk_states = query * key
    qk_states = query_states * key_states  # [num_layers, num_heads, head_dim//2]
    
    # If GQA, aggregate over query heads per kv head (matching original)
    if group_size > 1:
        qk_states = qk_states.reshape(
            num_layers, num_kv_heads, group_size, head_dim // 2
        ).sum(dim=2, keepdim=False)  # [num_layers, num_kv_heads, head_dim//2]

    # Sort by importance (descending) and assign ranks
    _, sorted_indices = torch.sort(qk_states, dim=-1, descending=True)  # [num_layers, num_kv_heads, head_dim//2]
    ranks = torch.empty_like(sorted_indices, dtype=torch.uint8)
    rank_values = torch.arange(qk_states.shape[-1], dtype=torch.uint8).expand_as(qk_states)
    ranks.scatter_(-1, sorted_indices, rank_values)

    # Duplicate for both halves of RoPE (real and imaginary parts use same ranking)
    ranks = torch.cat([ranks, ranks], dim=-1)  # [num_layers, num_kv_heads, head_dim]

    # Save
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        torch.save(ranks, f)
    
    print(f"Saved 2-norm rank tensor to {args.output_path}")
    print(f"Rank tensor shape: {ranks.shape} (expected: [num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}])")


if __name__ == "__main__":
    main()

