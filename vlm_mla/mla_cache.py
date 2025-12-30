from __future__ import annotations

from transformers.cache_utils import DynamicCache


class MLACache(DynamicCache):
    """
    A cache for MLA-style inference.

    We intentionally reuse the DynamicCache storage layout:
      - key_cache[layer]  : k_rope  (after RoPE), shape [bsz, n_kv_heads, seq_len, d_rope]
      - value_cache[layer]: c_kv    (latent),     shape [bsz, seq_len, d_kv_mid]

    The attention forward must know how to consume these tensors.
    """

    is_mla_cache: bool = True


