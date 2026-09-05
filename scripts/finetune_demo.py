"""Fine-tune TabPFN-M on block-missing data and compare with TabPFN.

python scripts/finetune_demo.py --epochs 30 --out results/finetune_friedman
"""
from __future__ import annotations

import argparse, json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from tabpfn_m import TabPFNMConfig, TabPFNMRegressor
from tabpfn_m.finetune import FinetunedTabPFNMRegressor, default_finetune_config
from benchmark_missing import inject_block

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--n-train", type=int, default=500)
ap.add_argument("--n-test", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--new-lr", type=float, default=1e-2)
ap.add_argument("--recon", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--source-shift", type=float, default=0.0,
                help="std of a per-source offset added to y, in units of std(y); makes the pattern informative")
ap.add_argument("--out", type=Path, default=Path("results/finetune_friedman"))
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

X, y = make_friedman1(n_samples=a.n_train + a.n_test, n_features=7, noise=0.5, random_state=a.seed)
# One block design over train+test so the same sources (and offsets) appear in both.
Xm, src = inject_block(X, np.random.default_rng(a.seed + 1), n_sources=4, keep=(3, 4), return_src=True)
if a.source_shift > 0:
    offsets = np.random.default_rng(a.seed + 3).normal(0, a.source_shift * y.std(), size=4)
    y = y + offsets[src]
    print("source offsets:", np.round(offsets, 3), flush=True)
Xtr, Xte, ytr, yte = X[: a.n_train], X[a.n_train :], y[: a.n_train], y[a.n_train :]
Xtr_m, Xte_m = Xm[: a.n_train], Xm[a.n_train :]
res = {}
def log(k, v):
    res[k] = float(v); print(f"{k:45s} R2 {v:.4f}", flush=True)

log("tabpfn_complete_train_complete_test", r2_score(yte, TabPFNRegressor(device="cpu", n_estimators=4, random_state=0).fit(Xtr, ytr).predict(Xte)))
log("tabpfn_block_train_block_test", r2_score(yte, TabPFNRegressor(device="cpu", n_estimators=4, random_state=0).fit(Xtr_m, ytr).predict(Xte_m)))
log("tabpfn_m_zero_shot_mask_bias1", r2_score(yte, TabPFNMRegressor(device="cpu", n_estimators=4, random_state=0, m_config=TabPFNMConfig(alpha_init=1.0, learn_alpha=False)).fit(Xtr_m, ytr).predict(Xte_m)))

common = dict(device="cpu", epochs=a.epochs, learning_rate=a.lr, n_finetune_ctx_plus_query_samples=250,
              n_estimators_finetune=1, n_estimators_validation=2, n_estimators_final_inference=4,
              validation_split_ratio=0.2, use_activation_checkpointing=False, early_stopping=True,
              early_stopping_patience=10, save_checkpoint_interval=None, random_state=a.seed, m_new_param_lr=a.new_lr)
for name, kw in [("ft_new_params_only", dict(m_freeze_base=True)), ("ft_full", dict(m_freeze_base=False))]:
    t = time.time()
    ft = FinetunedTabPFNMRegressor(m_config=default_finetune_config(recon_weight=a.recon), **common, **kw)
    ft.fit(Xtr_m, ytr)
    log(f"tabpfn_m_{name}", r2_score(yte, ft.predict(Xte_m)))
    print(f"  {time.time()-t:.0f}s alphas={np.round(ft.m_alphas(),2).tolist()} |absence|={float(ft._m_model().m_absence_embedding.norm()):.3f}", flush=True)
    ft.m_save(a.out / f"{name}.pt")
    res[f"{name}_alphas"] = ft.m_alphas()
    # Plain TabPFN fine-tuned the same way (no M components) as the fair control
if True:
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    ctl = {k: v for k, v in common.items() if k != "m_new_param_lr"}
    t = time.time()
    f0 = FinetunedTabPFNRegressor(**ctl); f0.fit(Xtr_m, ytr)
    log("tabpfn_plain_finetuned_control", r2_score(yte, f0.predict(Xte_m))); print(f"  {time.time()-t:.0f}s")
(a.out / "results.json").write_text(json.dumps(res, indent=2))
