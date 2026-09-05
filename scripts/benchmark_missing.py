"""Benchmark TabPFN vs TabPFN-M under structured missingness.

Designs
  block    rows come from K pseudo-sources; each source observes a fixed subset of
           3-4 out of F features (the user's setting: 7 features, some rows have
           3, some 4)
  mcar     cells missing completely at random at a given rate
  none     complete data (sanity: TabPFN-M must equal TabPFN)

Methods
  tabpfn        pretrained TabPFN, NaN passed through (its own indicator/imputation)
  tabpfn_mean   pretrained TabPFN on mean-imputed data (no indicator)
  hgb           sklearn HistGradientBoosting, native NaN support
  m_mask        TabPFN-M zero-shot: observed-only feature attention only
  m_mask_bias   TabPFN-M zero-shot: + pattern bias alpha=1 (test rows only)
  m_ft          TabPFN-M fine-tuned weights (if --weights given)

Usage
  python scripts/benchmark_missing.py --out results/zero_shot --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_diabetes, make_friedman1  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn_m import TabPFNMConfig, TabPFNMRegressor  # noqa: E402

COMPACTION = Path.home() / "env1/paper/compaction/paper22_aug/data.csv"


# ------------------------------------------------------------------ datasets ---
def load_datasets(max_rows: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    X, y = make_friedman1(n_samples=max_rows, n_features=7, noise=0.5, random_state=seed)
    out["friedman1_7f"] = (X, y)
    d = load_diabetes()
    out["diabetes"] = (d.data, d.target)
    if COMPACTION.exists():
        df = pd.read_csv(COMPACTION)
        feats = ["LL", "PL", "PI", "fines_pct", "sand_pct", "energy_kJm3", "Gs"]
        sub = df.dropna(subset=feats)  # complete rows only; missingness is injected
        if len(sub) > max_rows:
            sub = sub.sample(max_rows, random_state=seed)
        out["compaction_MDD"] = (sub[feats].to_numpy(float), sub["MDD_Mgm3"].to_numpy(float))
    return out


# --------------------------------------------------------------- missingness ---
def inject_block(X: np.ndarray, rng: np.random.Generator, n_sources: int, keep: tuple[int, int],
                 return_src: bool = False):
    """Hide features by pseudo-source. Each source observes a fixed subset of
    ``keep[0]``..``keep[1]`` features."""
    n, f = X.shape
    src = rng.integers(0, n_sources, size=n)
    subsets = []
    for _ in range(n_sources):
        k = rng.integers(keep[0], keep[1] + 1)
        subsets.append(set(rng.choice(f, size=min(k, f), replace=False).tolist()))
    Xm = X.copy()
    for i in range(n):
        hide = [j for j in range(f) if j not in subsets[src[i]]]
        Xm[i, hide] = np.nan
    return (Xm, src) if return_src else Xm


def inject_mcar(X: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    Xm = X.copy()
    m = rng.random(X.shape) < rate
    # keep at least one feature per row
    for i in np.where(m.all(1))[0]:
        m[i, rng.integers(0, X.shape[1])] = False
    Xm[m] = np.nan
    return Xm


DESIGNS = {
    "none": lambda X, rng: X.copy(),
    "mcar30": lambda X, rng: inject_mcar(X, rng, 0.30),
    "mcar50": lambda X, rng: inject_mcar(X, rng, 0.50),
    "block3-4of7": lambda X, rng: inject_block(X, rng, n_sources=4, keep=(3, 4)),
    "block2-5": lambda X, rng: inject_block(X, rng, n_sources=6, keep=(2, 5)),
}


# ------------------------------------------------------------------- methods ---
def mean_impute(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    f = lambda A: np.where(np.isnan(A), mu[None, :], A)
    return f(Xtr), f(Xte)


def run_method(name: str, Xtr, ytr, Xte, n_est: int, seed: int, weights: str | None):
    if name == "hgb":
        est = HistGradientBoostingRegressor(random_state=seed)
        est.fit(Xtr, ytr)
        return est.predict(Xte)
    if name == "tabpfn":
        est = TabPFNRegressor(device="cpu", n_estimators=n_est, random_state=seed)
        est.fit(Xtr, ytr)
        return est.predict(Xte)
    if name == "tabpfn_mean":
        a, b = mean_impute(Xtr, Xte)
        est = TabPFNRegressor(device="cpu", n_estimators=n_est, random_state=seed)
        est.fit(a, ytr)
        return est.predict(b)
    cfgs = {
        "m_mask": TabPFNMConfig(feature_mask=True, absence_embedding=False, pattern_bias=False, learn_alpha=False),
        "m_mask_bias": TabPFNMConfig(feature_mask=True, absence_embedding=False, pattern_bias=True, alpha_init=1.0, learn_alpha=False),
        "m_bias": TabPFNMConfig(feature_mask=False, absence_embedding=False, pattern_bias=True, alpha_init=1.0, learn_alpha=False),
        "m_ft": TabPFNMConfig(feature_mask=True, absence_embedding=True, pattern_bias=True, alpha_init=0.0, learn_alpha=True),
    }
    cfg = cfgs[name]
    est = TabPFNMRegressor(
        device="cpu", n_estimators=n_est, random_state=seed, m_config=cfg,
        m_weights=weights if name == "m_ft" else None,
    )
    est.fit(Xtr, ytr)
    return est.predict(Xte)


# ---------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/zero_shot"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--methods", nargs="+", default=["hgb", "tabpfn", "tabpfn_mean", "m_mask", "m_mask_bias"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--max-rows", type=int, default=600)
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--weights", type=str, default=None, help="fine-tuned TabPFN-M state dict for m_ft")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    rows = []
    csv_path = args.out / "results.csv"
    for seed in args.seeds:
        datasets = load_datasets(args.max_rows, seed)
        if args.datasets:
            datasets = {k: v for k, v in datasets.items() if k in args.datasets}
        for dname, (X, y) in datasets.items():
            Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
            for design in args.designs:
                rng = np.random.default_rng(1000 * seed + hash(design) % 997)
                Xall = DESIGNS[design](np.vstack([Xtr0, Xte0]), rng)
                Xtr, Xte = Xall[: len(Xtr0)], Xall[len(Xtr0):]
                for method in args.methods:
                    t0 = time.time()
                    try:
                        pred = run_method(method, Xtr, ytr, Xte, args.n_estimators, seed, args.weights)
                        r2 = r2_score(yte, pred)
                        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
                    except Exception as e:  # noqa: BLE001
                        print(f"FAILED {dname} {design} {method}: {e}")
                        r2, rmse = np.nan, np.nan
                    rows.append(dict(seed=seed, dataset=dname, design=design, method=method, r2=r2, rmse=rmse,
                                     missing_frac=float(np.isnan(Xall).mean()), secs=time.time() - t0))
                    print(f"seed={seed} {dname:16s} {design:12s} {method:12s} R2={r2:7.4f} RMSE={rmse:8.4f} ({rows[-1]['secs']:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby(["dataset", "design", "method"]).agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                                             rmse_mean=("rmse", "mean"), n=("r2", "size")).reset_index()
    summ.to_csv(args.out / "summary.csv", index=False)
    lines = ["# Missingness benchmark", "", f"seeds: {args.seeds}, n_estimators: {args.n_estimators}, max_rows: {args.max_rows}", ""]
    for (dname, design), g in summ.groupby(["dataset", "design"]):
        lines.append(f"## {dname} / {design}")
        lines.append("")
        lines.append("| method | R2 mean | R2 std | RMSE mean |")
        lines.append("|---|---|---|---|")
        for _, r in g.sort_values("r2_mean", ascending=False).iterrows():
            lines.append(f"| {r.method} | {r.r2_mean:.4f} | {r.r2_std:.4f} | {r.rmse_mean:.4f} |")
        lines.append("")
    (args.out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
