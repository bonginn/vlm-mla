from __future__ import annotations

import argparse

import torch
from PIL import Image

from .load import load_llava_with_optional_mla
from .mla_cache import MLACache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--prompt", required=True, type=str)
    p.add_argument("--max_new_tokens", default=128, type=int)
    p.add_argument("--dtype", default="bfloat16", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--revision", default=None, type=str)
    p.add_argument("--use_mla_cache", action="store_true", help="Use MLA latent cache for decoding (requires MLA model).")
    args = p.parse_args()

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    repo_root = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
    repo_root = __import__("os").path.dirname(repo_root)

    model, processor = load_llava_with_optional_mla(
        repo_root=repo_root,
        model_name_or_path=args.model,
        torch_dtype=torch_dtype,
        device=args.device,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        revision=args.revision,
    )

    image = Image.open(args.image).convert("RGB")

    # LLaVA expects the prompt to contain the image token(s). The most robust way
    # is to use the processor chat template.
    if hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        prompt = args.prompt if "<image>" in args.prompt else "<image>\n" + args.prompt

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            **({"past_key_values": MLACache()} if args.use_mla_cache else {}),
        )

    # Prefer processor.decode if exists; otherwise fallback to tokenizer
    gen_ids = out[0]
    if input_len is not None:
        gen_ids = gen_ids[input_len:]
    if hasattr(processor, "decode"):
        text = processor.decode(gen_ids, skip_special_tokens=True)
    else:
        text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()


