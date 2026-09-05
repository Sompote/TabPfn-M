"""Missingness-aware wrappers around TabPFN's MultiHeadAttention.

Both wrappers reuse the pretrained projection weights of the wrapped module and
only change how the attention scores are formed:

* ``MissingAwareFeatureAttention`` masks tokens of fully missing feature groups so
  they cannot be attended to as keys (observed-only feature attention).
* ``PatternBiasedItemAttention`` adds ``alpha * Jaccard(m_i, m_j)`` to the scores of
  the row-wise attention so rows with overlapping observed-feature sets inform each
  other more.

When the shared context is ``None`` (inference on complete data, or the component
switched off) the wrappers delegate to the wrapped module unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

from tabpfn_m.context import ContextHolder


def _qkv_from_inner(
    inner: MultiHeadAttention,
    x: torch.Tensor,
    x_kv: torch.Tensor | None,
    *,
    reuse_first_head_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project q, k, v with the wrapped module's weights. Shapes (B', S, H, D)."""
    q, k, v, kv, qkv = inner.compute_qkv(
        x,
        x_kv,
        None,
        None,
        None,
        cache_kv=False,
        use_cached_kv=False,
        reuse_first_head_kv=reuse_first_head_kv,
    )
    if qkv is not None:
        q, k, v = qkv.unbind(dim=-3)
    elif kv is not None:
        k, v = kv.unbind(dim=-3)
    assert q is not None and k is not None and v is not None
    return q, k, v


def _attend(
    q_BSHD: torch.Tensor,
    k_BSHD: torch.Tensor,
    v_BSHD: torch.Tensor,
    attn_mask: torch.Tensor | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Scaled dot-product attention with a boolean or additive mask."""
    q = q_BSHD.transpose(1, 2)
    k = k_BSHD.transpose(1, 2)
    v = v_BSHD.transpose(1, 2)
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    kwargs: dict[str, Any] = {}
    if softmax_scale is not None:
        kwargs["scale"] = softmax_scale
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)
    return out.transpose(1, 2)


class _WrapperBase(nn.Module):
    def __init__(self, inner: MultiHeadAttention, holder: ContextHolder) -> None:
        super().__init__()
        self.inner = inner
        # Plain attribute, not a submodule: the holder is shared, not owned.
        object.__setattr__(self, "holder", holder)

    # --- attributes PerFeatureEncoderLayer touches on the attention modules -----
    @property
    def has_cached_kv(self) -> bool:
        return self.inner.has_cached_kv

    def empty_kv_cache(self) -> None:
        self.inner.empty_kv_cache()

    def _project_out(self, heads_BSHD: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... h d, h d s -> ... s", heads_BSHD, self.inner._w_out)


class MissingAwareFeatureAttention(_WrapperBase):
    """Attention between the feature tokens of one row, with missing keys masked."""

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        cache_kv: bool = False,
        use_cached_kv: bool = False,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
    ) -> torch.Tensor:
        ctx = self.holder.ctx
        if ctx is None or not self.holder.feature_mask or x_kv is not None:
            return self.inner(
                x,
                x_kv,
                add_input=add_input,
                allow_inplace=allow_inplace,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_kv,
                use_cached_kv=use_cached_kv,
                reuse_first_head_kv=reuse_first_head_kv,
                only_cache_first_head_kv=only_cache_first_head_kv,
            )
        assert not (cache_kv or use_cached_kv), "KV caching is not supported by TabPFN-M."
        # x: (B, R, C, E)
        B, R, C, E = x.shape
        mask_BRC = ctx.token_mask_BRC
        assert mask_BRC.shape == (B, R, C), (mask_BRC.shape, x.shape)
        xf = x.reshape(B * R, C, E)
        q, k, v = _qkv_from_inner(self.inner, xf, None, reuse_first_head_kv=False)
        # Key mask broadcast over heads and queries: (B*R, 1, 1, C).
        attn_mask = mask_BRC.reshape(B * R, 1, 1, C)
        heads = _attend(q, k, v, attn_mask, self.inner.softmax_scale)
        out = self._project_out(heads).reshape(B, R, C, E)
        return x + out if add_input else out


class PatternBiasedItemAttention(_WrapperBase):
    """Attention between the rows of one column, biased by observed-set overlap."""

    def __init__(
        self,
        inner: MultiHeadAttention,
        holder: ContextHolder,
        *,
        alpha_init: float = 0.0,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__(inner, holder)
        alpha = torch.tensor(float(alpha_init))
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def _bias(
        self, ctx: Any, B: int, C: int, Rq: int, Rk: int, x_kv_given: bool
    ) -> torch.Tensor | None:
        R = ctx.num_rows
        sep = ctx.single_eval_pos
        if not x_kv_given and Rq == R:
            q0, k0 = 0, 0  # full self attention, keys = all rows
        elif x_kv_given and Rk == sep and Rq == R - sep:
            q0, k0 = sep, 0  # test rows attend to thinking + train rows
        elif x_kv_given and Rk == sep and Rq == sep:
            if self.holder.bias_test_only:
                return None
            q0, k0 = 0, 0  # thinking + train rows attend among themselves
        else:
            raise RuntimeError(
                f"Unrecognised item-attention slice: Rq={Rq}, Rk={Rk}, R={R}, sep={sep}"
            )
        sim = ctx.sim_BRR[:, q0 : q0 + Rq, k0 : k0 + Rk]  # (B, Rq, Rk)
        bias = self.alpha * sim
        # Same bias for every column and head: (B, C, 1, Rq, Rk) -> (B*C, 1, Rq, Rk)
        return bias[:, None, None].expand(B, C, 1, Rq, Rk).reshape(B * C, 1, Rq, Rk)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        cache_kv: bool = False,
        use_cached_kv: bool = False,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
    ) -> torch.Tensor:
        ctx = self.holder.ctx
        if ctx is None or not self.holder.pattern_bias:
            return self.inner(
                x,
                x_kv,
                add_input=add_input,
                allow_inplace=allow_inplace,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_kv,
                use_cached_kv=use_cached_kv,
                reuse_first_head_kv=reuse_first_head_kv,
                only_cache_first_head_kv=only_cache_first_head_kv,
            )
        assert not (cache_kv or use_cached_kv), "KV caching is not supported by TabPFN-M."
        # x: (B, C, Rq, E); x_kv: (B, C, Rk, E) or None
        B, C, Rq, E = x.shape
        Rk = x_kv.shape[2] if x_kv is not None else Rq
        xf = x.reshape(B * C, Rq, E)
        xkvf = x_kv.reshape(B * C, Rk, E) if x_kv is not None else None
        q, k, v = _qkv_from_inner(
            self.inner, xf, xkvf, reuse_first_head_kv=reuse_first_head_kv
        )
        bias = self._bias(ctx, B, C, Rq, Rk, x_kv is not None)
        heads = _attend(q, k, v, bias, self.inner.softmax_scale)
        out = self._project_out(heads).reshape(B, C, Rq, E)
        return x + out if add_input else out
