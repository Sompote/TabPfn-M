"""TabPFN vs TabPFN-M on the compaction database with source-structured folds.

Designs
  real          the data as merged: sand_pct absent for one source, LL/PL 7% missing in LTPP
  source_block  on top of real: every real source hides a fixed random subset of the
                7 features so each row keeps 3-4 (the user's setting, with real source effects)
Folds
  loso          leave one of the 6 sources out
  group5        GroupKFold(5) over the 162 provenance groups
Targets: MDD_Mgm3, OMC_frac.

python scripts/compaction_loso.py --out results/compaction_zero_shot
python scripts/compaction_loso.py --finetune --designs real source_block --folds loso --out results/compaction_finetune
"""
from __future__ import annotations

import argparse, json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor
from tabpfn_m import TabPFNMConfig, TabPFNMRegressor

DATA = Path.home() / "env1/paper/compaction/paper22_aug/data.csv"
FEATS = ["LL", "PL", "PI", "fines_pct", "sand_pct", "energy_kJm3", "Gs"]
TARGETS = ["MDD_Mgm3", "OMC_frac"]


def source_block(X: np.ndarray, src: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    for s in np.unique(src):
        k = rng.integers(3, 5)
        keep = set(rng.choice(X.shape[1], size=k, replace=False).tolist())
        hide = [j for j in range(X.shape[1]) if j not in keep]
        Xm[np.ix_(src == s, hide)] = np.nan
    return Xm


def folds(kind: str, src: np.ndarray, grp: np.ndarray):
    if kind == "loso":
        for s in np.unique(src):
            yield s, np.where(src != s)[0], np.where(src == s)[0]
    else:
        for i, (tr, te) in enumerate(GroupKFold(5).split(src, groups=grp)):
            yield f"fold{i}", tr, te


def fit_predict(method, Xtr, ytr, Xte, n_est, seed, ft_kw):
    if method == "hgb":
        return HistGradientBoostingRegressor(random_state=seed).fit(Xtr, ytr).predict(Xte)
    if method == "tabpfn":
        return TabPFNRegressor(device="cpu", n_estimators=n_est, random_state=seed, ignore_pretraining_limits=True).fit(Xtr, ytr).predict(Xte)
    if method == "m_mask":
        cfg = TabPFNMConfig(feature_mask=True, absence_embedding=False, pattern_bias=False, learn_alpha=False)
    elif method == "m_mask_bias":
        cfg = TabPFNMConfig(feature_mask=True, absence_embedding=False, pattern_bias=True, alpha_init=1.0, learn_alpha=False)
    elif method == "tabpfn_ft":
        from tabpfn.finetuning import FinetunedTabPFNRegressor
        return FinetunedTabPFNRegressor(**ft_kw).fit(Xtr, ytr).predict(Xte)
    elif method == "m_ft":
        from tabpfn_m.finetune import FinetunedTabPFNMRegressor, default_finetune_config
        est = FinetunedTabPFNMRegressor(m_config=default_finetune_config(recon_weight=0.1), m_new_param_lr=1e-2, **ft_kw)
        est.fit(Xtr, ytr)
        print("      alphas:", np.round(est.m_alphas(), 2).tolist(), flush=True)
        return est.predict(Xte)
    else:
        raise ValueError(method)
    return TabPFNMRegressor(device="cpu", n_estimators=n_est, random_state=seed, m_config=cfg, ignore_pretraining_limits=True).fit(Xtr, ytr).predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/compaction_zero_shot"))
    ap.add_argument("--designs", nargs="+", default=["real", "source_block"])
    ap.add_argument("--folds", nargs="+", default=["loso", "group5"])
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--finetune", action="store_true", help="run tabpfn_ft and m_ft instead of zero-shot methods")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    methods = a.methods or (["tabpfn_ft", "m_ft"] if a.finetune else ["hgb", "tabpfn", "m_mask", "m_mask_bias"])
    ft_kw = dict(device="cpu", epochs=a.epochs, learning_rate=1e-5, n_finetune_ctx_plus_query_samples=1000,
                 n_estimators_finetune=1, n_estimators_validation=2, n_estimators_final_inference=a.n_estimators,
                 validation_split_ratio=0.15, use_activation_checkpointing=False, early_stopping=True,
                 early_stopping_patience=4, save_checkpoint_interval=None, random_state=a.seed)

    df = pd.read_csv(DATA)
    src = df["source"].to_numpy(); grp = df["group"].to_numpy()
    X_real = df[FEATS].to_numpy(float)
    print("rows", len(df), "sources", {s: int((src == s).sum()) for s in np.unique(src)}, flush=True)
    print("missing frac per source (real):"); print(pd.DataFrame(np.isnan(X_real), columns=FEATS).groupby(src).mean().round(2).to_string(), flush=True)

    rows = []
    for design in a.designs:
        X = X_real if design == "real" else source_block(X_real, src, a.seed)
        for target in a.targets:
            y = df[target].to_numpy(float)
            for fold_kind in a.folds:
                for name, tr, te in folds(fold_kind, src, grp):
                    for method in methods:
                        t0 = time.time()
                        try:
                            pred = fit_predict(method, X[tr], y[tr], X[te], a.n_estimators, a.seed, ft_kw)
                            r2 = r2_score(y[te], pred); rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
                        except Exception as e:  # noqa: BLE001
                            print(f"FAILED {design} {target} {fold_kind} {name} {method}: {e}", flush=True); r2 = rmse = np.nan
                        rows.append(dict(design=design, target=target, folds=fold_kind, fold=str(name), method=method,
                                         n_test=len(te), r2=r2, rmse=rmse, secs=time.time() - t0))
                        print(f"{design:12s} {target:9s} {fold_kind:6s} {str(name)[:28]:28s} {method:12s} R2={r2:7.4f} RMSE={rmse:.4f} n={len(te)} ({rows[-1]['secs']:.0f}s)", flush=True)
                        pd.DataFrame(rows).to_csv(a.out / "results.csv", index=False)
    res = pd.DataFrame(rows)
    # pooled R2 per (design, target, folds, method): weight folds by test size via pooled predictions is not
    # available, so report mean R2 over folds and the R2 of the pooled RMSE proxy.
    summ = res.groupby(["design", "target", "folds", "method"]).agg(r2_mean=("r2", "mean"), r2_median=("r2", "median"),
                                                                       rmse_mean=("rmse", "mean"), n_folds=("r2", "size")).reset_index()
    summ.to_csv(a.out / "summary.csv", index=False)
    lines = [f"# Compaction database, source-structured folds (seed {a.seed}, n_estimators {a.n_estimators})", ""]
    for (d, t, f), g in summ.groupby(["design", "target", "folds"]):
        lines += [f"## {d} / {t} / {f}", "", "| method | R2 mean over folds | R2 median | RMSE mean |", "|---|---|---|---|"]
        lines += [f"| {r.method} | {r.r2_mean:.4f} | {r.r2_median:.4f} | {r.rmse_mean:.4f} |" for _, r in g.sort_values("r2_mean", ascending=False).iterrows()]
        lines.append("")
    (a.out / "summary.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
