"""Block-structured missingness augmentation for fine-tuning."""

from __future__ import annotations

import torch

from tabpfn_m.config import AugmentConfig


def block_missingness(
    x_SBF: torch.Tensor,
    cfg: AugmentConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide feature blocks per pseudo-source, plus a light MCAR sprinkle.

    Every batch element (one in-context dataset) is treated independently. Its rows
    are assigned to ``n_sources`` pseudo-sources; each pseudo-source keeps a random
    subset of at least one feature and hides the others. This produces the
    row-block structure seen when tables are merged from laboratories that ran
    different test programmes.

    Returns:
        (x_aug, hidden) where ``hidden`` is True for cells that were observed in
        ``x_SBF`` and are NaN in ``x_aug``.
    """
    S, B, F = x_SBF.shape
    dev = x_SBF.device
    g = generator

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, device=dev)

    hide = torch.zeros(S, B, F, dtype=torch.bool, device=dev)
    for b in range(B):
        if rand(1).item() > cfg.p_apply:
            continue
        n_src = int(
            torch.randint(
                cfg.n_sources_min, cfg.n_sources_max + 1, (1,), generator=g
            ).item()
        )
        # Row -> pseudo-source. Contiguous blocks after a random permutation are not
        # needed: TabPFN is permutation invariant over rows.
        src_of_row = torch.randint(0, n_src, (S,), generator=g, device=dev)
        for s in range(n_src):
            keep_frac = (
                cfg.keep_frac_min
                + (cfg.keep_frac_max - cfg.keep_frac_min) * rand(1).item()
            )
            n_keep = max(1, int(round(keep_frac * F)))
            perm = torch.randperm(F, generator=g, device=dev)
            hidden_feats = perm[n_keep:]
            row_idx = (src_of_row == s).nonzero(as_tuple=True)[0]
            if hidden_feats.numel() and row_idx.numel():
                hide[row_idx[:, None], b, hidden_feats[None, :]] = True
        if cfg.p_mcar > 0:
            hide[:, b] |= rand(S, F) < cfg.p_mcar
    observed = torch.isfinite(x_SBF)
    hidden = hide & observed
    # Never hide every feature of a row: keep one observed cell at random.
    all_hidden = (observed & ~hidden).sum(-1) == 0
    if all_hidden.any():
        idx = all_hidden.nonzero(as_tuple=False)
        for s_i, b_i in idx.tolist():
            obs_feats = observed[s_i, b_i].nonzero(as_tuple=True)[0]
            if obs_feats.numel():
                j = obs_feats[torch.randint(0, obs_feats.numel(), (1,), generator=g)]
                hidden[s_i, b_i, j] = False
    x_aug = x_SBF.clone()
    x_aug[hidden] = float("nan")
    return x_aug, hidden


def reconstruction_mask(
    x_SBF: torch.Tensor,
    frac: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cell-wise mask over observed cells for the reconstruction objective."""
    observed = torch.isfinite(x_SBF)
    r = torch.rand(x_SBF.shape, generator=generator, device=x_SBF.device)
    m = observed & (r < frac)
    # Keep at least one observed cell per row.
    remaining = (observed & ~m).sum(-1)
    fix = (remaining == 0) & observed.any(-1)
    if fix.any():
        for s_i, b_i in fix.nonzero(as_tuple=False).tolist():
            j = m[s_i, b_i].nonzero(as_tuple=True)[0][0]
            m[s_i, b_i, j] = False
    return m


def standardise_targets(
    x_SBF: torch.Tensor, num_train_rows: int
) -> torch.Tensor:
    """Standardise each column with the mean/std of its observed train-row values."""
    train = x_SBF[:num_train_rows]
    obs = torch.isfinite(train)
    cnt = obs.sum(0).clamp_min(1)
    mean = torch.where(obs, train, torch.zeros_like(train)).sum(0) / cnt
    var = (torch.where(obs, train - mean, torch.zeros_like(train)) ** 2).sum(0) / cnt
    std = var.sqrt().clamp_min(1e-6)
    z = (x_SBF - mean[None]) / std[None]
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10, 10)
