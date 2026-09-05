"""Tests for TabPFN-M. Run with: python -m pytest tests -q"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn import TabPFNRegressor
from tabpfn_m import AugmentConfig, TabPFNMConfig, TabPFNMRegressor, upgrade_model
from tabpfn_m.augment import block_missingness, reconstruction_mask, standardise_targets
from tabpfn_m.context import build_context, jaccard_similarity


def _block_data(n=240, f=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + X[:, 2] * X[:, 3] + 0.1 * rng.normal(size=n)
    src = rng.integers(0, 3, size=n)
    keep = {0: [0, 1, 2], 1: [0, 3, 4, 5], 2: [1, 2, 6]}
    Xm = X.copy()
    for i in range(n):
        Xm[i, [j for j in range(f) if j not in keep[src[i]]]] = np.nan
    return X, Xm, y


@pytest.fixture(scope="module")
def fitted():
    X, Xm, y = _block_data()
    reg = TabPFNRegressor(device="cpu", n_estimators=1, random_state=0)
    reg.fit(Xm[:180], y[:180])
    return reg, X, Xm, y


# ----------------------------------------------------------------- context ---
def test_jaccard():
    obs = torch.tensor([[[1, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=torch.bool)
    s = jaccard_similarity(obs)[0]
    assert torch.allclose(s.diag(), torch.ones(3))
    assert s[0, 1] == pytest.approx(0.5)
    assert s[0, 2] == pytest.approx(0.0)


def test_build_context_shapes_and_alignment():
    x = torch.randn(5, 2, 7)
    x[0, 0, 3:] = float("nan")  # row 0 of batch 0: groups 1 (feats 3-5) and 2 (feat 6) missing
    ctx = build_context(x, features_per_group=3, num_thinking_rows=4, num_train_rows=3)
    assert ctx.token_mask_BRC.shape == (2, 9, 4)  # 3 groups + target
    assert ctx.sim_BRR.shape == (2, 9, 9)
    assert ctx.group_missing_BRG[0, 4, :].tolist() == [False, True, True]
    assert ctx.token_mask_BRC[0, 4, :].tolist() == [True, False, False, True]
    assert ctx.token_mask_BRC[:, :4].all()  # thinking rows always attendable
    assert torch.allclose(ctx.sim_BRR[0, :4], torch.ones(4, 9))  # thinking queries uniform


def test_context_complete_data_is_all_ones():
    ctx = build_context(torch.randn(6, 1, 4), features_per_group=3, num_thinking_rows=2, num_train_rows=4)
    assert ctx.token_mask_BRC.all()
    assert torch.allclose(ctx.sim_BRR, torch.ones_like(ctx.sim_BRR))


# ------------------------------------------------------------ equivalence ---
def test_all_off_matches_pretrained(fitted):
    reg, X, Xm, y = fitted
    p0 = reg.predict(Xm[180:])
    upgrade_model(reg.models_[0], TabPFNMConfig.baseline())
    p1 = reg.predict(Xm[180:])
    assert np.array_equal(p0, p1)


def test_complete_data_matches_pretrained_with_components_on():
    X, Xm, y = _block_data()
    reg = TabPFNRegressor(device="cpu", n_estimators=1, random_state=0)
    reg.fit(X[:180], y[:180])
    p0 = reg.predict(X[180:])
    upgrade_model(reg.models_[0], TabPFNMConfig(alpha_init=3.0, learn_alpha=False, bias_test_only=False))
    p1 = reg.predict(X[180:])
    assert np.abs(p0 - p1).max() < 1e-3


def test_components_change_predictions_on_missing_data(fitted):
    reg, X, Xm, y = fitted
    upgrade_model(reg.models_[0], TabPFNMConfig.baseline())
    p0 = reg.predict(Xm[180:])
    upgrade_model(reg.models_[0], TabPFNMConfig(alpha_init=1.0, learn_alpha=False))
    p1 = reg.predict(Xm[180:])
    assert np.isfinite(p1).all()
    assert np.abs(p0 - p1).max() > 1e-4
    r2 = lambda p: 1 - ((p - y[180:]) ** 2).sum() / ((y[180:] - y[180:].mean()) ** 2).sum()
    assert r2(p1) > 0.0


def test_reupgrade_resets_alpha(fitted):
    reg, *_ = fitted
    m = upgrade_model(reg.models_[0], TabPFNMConfig(alpha_init=2.0, learn_alpha=False))
    assert m.m_alphas()[0] == pytest.approx(2.0)
    upgrade_model(m, TabPFNMConfig(alpha_init=0.5, learn_alpha=True))
    assert m.m_alphas()[0] == pytest.approx(0.5)
    assert any(p.requires_grad for p in m.m_new_parameters())


# --------------------------------------------------- observed-only attention ---
def test_feature_mask_blocks_gradient_from_missing_tokens(fitted):
    """With the mask on, a test row's prediction must not depend on the embedded
    tokens of its fully missing feature groups."""
    reg, X, Xm, y = fitted
    m = upgrade_model(
        reg.models_[0],
        TabPFNMConfig(feature_mask=True, absence_embedding=False, pattern_bias=False),
    )
    m.eval()
    x = torch.tensor(Xm[:60], dtype=torch.float32)[:, None, :]
    yt = torch.tensor(y[:40], dtype=torch.float32)[:, None]
    grads = {}

    def capture(mod, inp, out):
        out.retain_grad()
        grads["emb"] = out

    h = m.encoder.register_forward_hook(capture)
    try:
        for on in (True, False):
            m.m_set_active(feature_mask=on)
            with torch.enable_grad():
                out = m(x.clone(), yt.clone())
                out[:, :, :].sum().backward()
            g = grads["emb"].grad  # (s, b*G, e)
            G = g.shape[1]
            g = g.reshape(60, 1, G, -1)
            ctx = build_context(x, features_per_group=m.features_per_group, num_thinking_rows=0, num_train_rows=40)
            miss = ctx.group_missing_BRG[0]  # (s, G)
            test_missing_grad = g[40:, 0][miss[40:]].abs().max().item()
            if on:
                assert test_missing_grad == 0.0
            else:
                assert test_missing_grad > 0.0
    finally:
        h.remove()
        m.m_set_active(feature_mask=True)


# --------------------------------------------------------------- augmenter ---
def test_block_missingness_structure():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 1, 7)
    cfg = AugmentConfig(p_apply=1.0, n_sources_min=3, n_sources_max=3, p_mcar=0.0, keep_frac_min=0.4, keep_frac_max=0.6)
    xa, hidden = block_missingness(x, cfg, g)
    assert hidden.any()
    assert torch.isnan(xa[hidden]).all()
    assert not torch.isnan(xa[~hidden]).any()
    patterns = {tuple(r.tolist()) for r in torch.isnan(xa[:, 0]).int()}
    assert len(patterns) <= 3  # one pattern per pseudo-source
    assert (torch.isfinite(xa).sum(-1) >= 1).all()  # every row keeps a feature


def test_reconstruction_mask_and_targets():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(100, 2, 5)
    x[0, 0, 0] = float("nan")
    m = reconstruction_mask(x, 0.2, g)
    assert not m[0, 0, 0]
    assert 0.1 < m.float().mean().item() < 0.3
    z = standardise_targets(x, 80)
    assert torch.isfinite(z).all()
    assert z[:80, 0, 1].mean().abs() < 0.2


# ------------------------------------------------------------- training path ---
def test_training_forward_with_augment_and_recon_backward(fitted):
    reg, X, Xm, y = fitted
    m = upgrade_model(
        reg.models_[0],
        TabPFNMConfig(alpha_init=0.0, learn_alpha=True, recon_weight=1.0, augment=AugmentConfig(p_apply=1.0)),
    )
    m.m_training = True
    try:
        x = torch.tensor(X[:80], dtype=torch.float32)[:, None, :]
        yt = torch.tensor(y[:60], dtype=torch.float32)[:, None]
        with torch.enable_grad():
            out = m(x, yt)
            aux = m.m_pop_aux_loss()
            assert aux is not None and torch.isfinite(aux)
            (out.sum() * 0 + aux).backward()
        assert m.m_recon_head.weight.grad is not None
        assert m.m_pop_aux_loss() is None
        with torch.no_grad():  # inference: augmentation must be off
            m(x, yt)
        assert m.m_pop_aux_loss() is None
    finally:
        m.m_training = False


def test_estimator_roundtrip():
    X, Xm, y = _block_data()
    est = TabPFNMRegressor(device="cpu", n_estimators=1, random_state=0, m_config=TabPFNMConfig(alpha_init=1.0, learn_alpha=False))
    est.fit(Xm[:180], y[:180])
    assert type(est.models_[0]).__name__ == "TabPFNMTransformer"
    p = est.predict(Xm[180:])
    assert p.shape == (60,) and np.isfinite(p).all()
    assert "m_config" in est.get_params()


# ---------------------------------------------------------------- finetuning ---
def test_param_group_optimizer_builds_two_groups(fitted):
    from tabpfn.finetuning import finetuned_base as fb
    from tabpfn_m.finetune import _param_group_optimizer

    reg, *_ = fitted
    m = upgrade_model(reg.models_[0], TabPFNMConfig(alpha_init=0.0, learn_alpha=True))
    with _param_group_optimizer(lambda: m, new_lr=1e-2, freeze_base=False):
        opt = fb.get_and_init_optimizer(model_parameters=m.parameters(), learning_rate=1e-5, weight_decay=0.01)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-2)
    assert len(opt.param_groups[1]["params"]) == len(m.m_new_parameters())
    with _param_group_optimizer(lambda: m, new_lr=1e-2, freeze_base=True):
        opt = fb.get_and_init_optimizer(model_parameters=m.parameters(), learning_rate=1e-5, weight_decay=0.01)
    assert len(opt.param_groups) == 1
    assert all(not p.requires_grad for p in m.parameters() if id(p) not in {id(q) for q in m.m_new_parameters()})
    for p in m.parameters():
        p.requires_grad_(True)


def test_classifier_with_missing_features():
    from tabpfn_m import TabPFNMClassifier

    X, Xm, y = _block_data()
    yc = (y > np.median(y)).astype(int)
    clf = TabPFNMClassifier(device="cpu", n_estimators=1, random_state=0, m_config=TabPFNMConfig(alpha_init=1.0, learn_alpha=False))
    clf.fit(Xm[:180], yc[:180])
    proba = clf.predict_proba(Xm[180:])
    assert proba.shape == (60, 2) and np.allclose(proba.sum(1), 1.0, atol=1e-4)
    assert (clf.predict(Xm[180:]) == yc[180:]).mean() > 0.55
