from __future__ import annotations

import inspect
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, rotate_half, repeat_kv

from .mla_cache import MLACache


def _create_custom_apply_rotary_pos_emb(q_r_indices, k_r_indices):
    """
    A transformers-4.46 compatible version of the paper's custom RoPE application:
    it applies RoPE only to the selected (reordered) q_r/k_r dimensions.
    """

    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Get the calling frame and the `self` of attention module to read layer_idx.
        frame = inspect.currentframe().f_back
        attention_module = frame.f_locals["self"]
        layer_idx = attention_module.layer_idx

        q_idx = q_r_indices[layer_idx].to(q.device)
        cos_q = cos.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        sin_q = sin.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        cos_q = cos_q.reshape(cos_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)
        sin_q = sin_q.reshape(sin_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)

        k_idx = k_r_indices[layer_idx].to(k.device)
        cos_k = cos.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        sin_k = sin.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        cos_k = cos_k.reshape(cos_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)
        sin_k = sin_k.reshape(sin_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_embed, k_embed

    return custom_apply_rotary_pos_emb


def _mla_forward_sdpa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """
    MLA forward for llama attention modules in transformers 4.46.x.
    Works for both LlamaAttention (manual) and LlamaSdpaAttention (SDPA) by
    using SDPA by default and optionally computing weights when requested.
    """
    bsz, q_len, _ = hidden_states.size()

    # We always compute Q and the RoPE-part K for the current tokens.
    query_states_full = self.q_proj(hidden_states)
    key_r_states = self.k_r_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)

    q_r_dim = self.num_heads * key_r_states.size(-1)
    query_r = query_states_full[..., :q_r_dim].view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
    query_nope = query_states_full[..., q_r_dim:].view(bsz, q_len, self.num_heads, -1).transpose(1, 2)

    # RoPE embeddings for the RoPE subspace only (we reuse the model's cos/sin).
    if position_embeddings is None:
        # Use any tensor with correct dtype/device to compute cos/sin
        dummy = torch.zeros(bsz, self.num_key_value_heads, q_len, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        cos, sin = self.rotary_emb(dummy, position_ids)
    else:
        cos, sin = position_embeddings
    query_r, key_r_states = modeling_llama.apply_rotary_pos_emb(query_r, key_r_states, cos, sin)

    # If the caller provided an MLA cache, cache (k_rope, c_kv) instead of full (K,V).
    if isinstance(past_key_value, MLACache) or getattr(past_key_value, "is_mla_cache", False):
        # latent for current tokens
        if not hasattr(self.kv_proj, "down_kv"):
            raise ValueError("MLA cache requires kv_proj.down_kv (joint low-rank).")
        c_kv = self.kv_proj.down_kv(hidden_states)  # [bsz, q_len, d_mid]

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k_rope_all, c_kv_all = past_key_value.update(key_r_states, c_kv, self.layer_idx, cache_kwargs)

        # Fast path: decoding (q_len==1) with chunked online softmax to avoid materializing full K/V.
        if q_len == 1:
            chunk_size = 256
            # prepare q parts
            q_rope = query_r  # [bsz, n_heads, 1, d_rope]
            q_nope = query_nope  # [bsz, n_heads, 1, d_nope] (may be 0-dim)
            d_nope = q_nope.size(-1)

            kv_len = k_rope_all.size(-2)
            # attention mask slice helper
            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, :kv_len]

            m = None  # running max
            l = None  # running exp sum
            o = None  # running weighted sum (in head_dim space)

            # We'll reconstruct v in chunks from c_kv_all and compute scores from (k_rope, k_nope).
            for start in range(0, kv_len, chunk_size):
                end = min(start + chunk_size, kv_len)

                k_rope_chunk = k_rope_all[:, :, start:end, :]  # [bsz, n_kv, chunk, d_rope]
                k_rope_chunk = repeat_kv(k_rope_chunk, self.num_key_value_groups)  # [bsz, n_heads, chunk, d_rope]

                c_chunk = c_kv_all[:, start:end, :]  # [bsz, chunk, d_mid]
                # v: [bsz, chunk, n_kv*head_dim] -> [bsz,n_kv,chunk,head_dim] -> repeat_kv
                v_chunk = self.kv_proj.up_v(c_chunk)
                v_chunk = v_chunk.view(bsz, end - start, self.num_key_value_heads, -1).transpose(1, 2)
                v_chunk = repeat_kv(v_chunk, self.num_key_value_groups)  # [bsz, n_heads, chunk, head_dim]

                if d_nope > 0:
                    k_nope_chunk = self.kv_proj.up_k(c_chunk)
                    k_nope_chunk = k_nope_chunk.view(bsz, end - start, self.num_key_value_heads, -1).transpose(1, 2)
                    k_nope_chunk = repeat_kv(k_nope_chunk, self.num_key_value_groups)  # [bsz,n_heads,chunk,d_nope]
                    score_nope = torch.matmul(q_nope, k_nope_chunk.transpose(2, 3))  # [bsz,n_heads,1,chunk]
                else:
                    score_nope = 0.0

                score_rope = torch.matmul(q_rope, k_rope_chunk.transpose(2, 3))  # [bsz,n_heads,1,chunk]
                scores = (score_rope + score_nope) / math.sqrt(self.head_dim)

                if causal_mask is not None:
                    scores = scores + causal_mask[:, :, :, start:end]

                # online softmax update
                scores_max = scores.amax(dim=-1, keepdim=True)  # [bsz,n_heads,1,1]
                if m is None:
                    m = scores_max
                    p = torch.exp(scores - m)
                    l = p.sum(dim=-1, keepdim=True)
                    o = torch.matmul(p, v_chunk)  # [bsz,n_heads,1,head_dim]
                else:
                    m_new = torch.maximum(m, scores_max)
                    alpha = torch.exp(m - m_new)
                    p = torch.exp(scores - m_new)
                    l = l * alpha + p.sum(dim=-1, keepdim=True)
                    o = o * alpha + torch.matmul(p, v_chunk)
                    m = m_new

            attn_output = o / l  # [bsz,n_heads,1,head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

        # Prefill path (q_len>1): materialize full K/V once (ephemeral), but cache stays latent.
        c_all = c_kv_all  # [bsz, kv_len, d_mid]
        v_all = self.kv_proj.up_v(c_all).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_all = repeat_kv(v_all, self.num_key_value_groups)
        d_nope = query_nope.size(-1)
        if d_nope > 0:
            k_nope_all = self.kv_proj.up_k(c_all).view(bsz, -1, self.num_key_value_heads, d_nope).transpose(1, 2)
            k_nope_all = repeat_kv(k_nope_all, self.num_key_value_groups)
        else:
            k_nope_all = None
        k_rope_all_rep = repeat_kv(k_rope_all, self.num_key_value_groups)

        q_rope = query_r
        q_nope = query_nope
        score_rope = torch.matmul(q_rope, k_rope_all_rep.transpose(2, 3))
        if k_nope_all is not None:
            score_nope = torch.matmul(q_nope, k_nope_all.transpose(2, 3))
        else:
            score_nope = 0.0
        attn_weights = (score_rope + score_nope) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = attn_weights + causal_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_probs, v_all)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, (attn_probs if output_attentions else None), past_key_value

    # Default path (no MLA cache): behave like our previous "patched MHA" attention and cache full K/V.
    key_c_states, value_states = self.kv_proj.mha_forward(hidden_states)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)
    key_c_states = key_c_states.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2)
    query_states = torch.cat([query_r, query_nope], dim=-1)
    key_states = torch.cat([key_r_states, key_c_states], dim=-1)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    is_causal = True if causal_mask is None and q_len > 1 else False
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def mha2mla_llama_compat(q_idx, k_idx):
    """
    Monkey patch llama attention for transformers 4.46.x.

    - Overrides RoPE application to only affect selected dimensions (partial-RoPE).
    - Overrides LlamaSdpaAttention.forward (and base LlamaAttention.forward) to use MLA projections.
    """
    modeling_llama.apply_rotary_pos_emb = _create_custom_apply_rotary_pos_emb(q_idx, k_idx)
    LlamaSdpaAttention.forward = _mla_forward_sdpa
    LlamaAttention.forward = _mla_forward_sdpa


