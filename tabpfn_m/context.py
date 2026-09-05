"""Missingness context shared between the model forward and the attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MissingnessContext:
    """Row/token level missingness information for one forward pass.

    Shapes use B = batch, R = thinking + train + test rows, C = feature groups + 1
    (the last column is the target), F = raw features.
    """

    token_mask_BRC: torch.Tensor
    """bool, True where a token may be attended to as a key in feature attention."""
    group_missing_BRG: torch.Tensor
    """bool, True where a feature group has no observed feature (thinking rows False)."""
    sim_BRR: torch.Tensor
    """float, Jaccard overlap of observed feature sets; 0 for thinking-row pairs."""
    single_eval_pos: int
    """Thinking + train rows: the key range of the item attention."""
    num_thinking_rows: int

    @property
    def num_rows(self) -> int:
        return self.token_mask_BRC.shape[1]


class ContextHolder:
    """Mutable pointer shared by all attention wrappers of one model."""

    def __init__(self) -> None:
        self.ctx: MissingnessContext | None = None
        self.feature_mask: bool = True
        self.pattern_bias: bool = True
        self.bias_test_only: bool = True


def observed_mask(x_SBF: torch.Tensor) -> torch.Tensor:
    """True where a raw cell carries a usable value (not NaN, not inf)."""
    return torch.isfinite(x_SBF)


def jaccard_similarity(obs_BSF: torch.Tensor) -> torch.Tensor:
    """Jaccard overlap of observed-feature sets between all row pairs.

    Args:
        obs_BSF: bool (B, S, F).

    Returns:
        float (B, S, S) in [0, 1]; pairs where both rows observe nothing get 0.
    """
    o = obs_BSF.to(torch.float32)
    inter = torch.einsum("bif,bjf->bij", o, o)
    n = o.sum(-1)
    union = n[:, :, None] + n[:, None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))


def build_context(
    x_SBF: torch.Tensor,
    *,
    features_per_group: int,
    num_thinking_rows: int,
    num_train_rows: int,
) -> MissingnessContext:
    """Build the missingness context from the raw (NaN-carrying) model input.

    The token layout follows PerFeatureTransformer: features are zero-padded to a
    multiple of ``features_per_group`` and grouped in order, the target column is
    appended last, and ``num_thinking_rows`` rows are prepended.
    """
    S, B, F = x_SBF.shape
    n = features_per_group
    obs_SBF = observed_mask(x_SBF)
    pad = (-F) % n
    if pad:
        # Padding columns are not real features: they neither count as observed nor
        # as missing.
        obs_pad = torch.zeros(S, B, pad, dtype=torch.bool, device=x_SBF.device)
        obs_grp = torch.cat([obs_SBF, obs_pad], dim=-1)
    else:
        obs_grp = obs_SBF
    G = obs_grp.shape[-1] // n
    obs_BSGn = obs_grp.permute(1, 0, 2).reshape(B, S, G, n)
    group_obs_any_BSG = obs_BSGn.any(-1)
    real_in_group = torch.ones(G, n, dtype=torch.bool, device=x_SBF.device).reshape(-1)
    if pad:
        real_in_group[F:] = False
    real_in_group = real_in_group.reshape(G, n).any(-1)  # (G,) group has a real feature
    # A group made only of padding columns is never "missing"; it is a constant zero
    # column which the encoder treats like any other token.
    group_missing_BSG = (~group_obs_any_BSG) & real_in_group[None, None, :]

    T = num_thinking_rows
    R = T + S
    device = x_SBF.device

    token_mask_BRC = torch.ones(B, R, G + 1, dtype=torch.bool, device=device)
    token_mask_BRC[:, T:, :G] = ~group_missing_BSG

    group_missing_BRG = torch.zeros(B, R, G, dtype=torch.bool, device=device)
    group_missing_BRG[:, T:] = group_missing_BSG

    sim_BSS = jaccard_similarity(obs_SBF.permute(1, 0, 2))
    # Thinking rows carry no missingness pattern. As keys they receive the query's
    # mean overlap with the training rows, as queries they see a uniform value, so
    # the bias is neutral for them. With complete data every entry is 1 and the
    # softmax is unchanged: the model then equals the pretrained TabPFN exactly.
    sim_BRR = torch.ones(B, R, R, dtype=torch.float32, device=device)
    sim_BRR[:, T:, T:] = sim_BSS
    if T > 0:
        n_train = max(num_train_rows, 1)
        mean_to_train = sim_BSS[:, :, :n_train].mean(-1, keepdim=True)  # (B, S, 1)
        sim_BRR[:, T:, :T] = mean_to_train.expand(B, S, T)

    return MissingnessContext(
        token_mask_BRC=token_mask_BRC,
        group_missing_BRG=group_missing_BRG,
        sim_BRR=sim_BRR,
        single_eval_pos=T + num_train_rows,
        num_thinking_rows=T,
    )
