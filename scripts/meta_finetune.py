"""Meta-fine-tune the TabPFN-M components across many synthetic block-missing tasks.

The new parameters (per-layer alpha, absence vector, reconstruction head) are
prior-level: they must be learned across tasks, not on one small dataset. This
script draws random regression tasks, applies block-missingness augmentation and
masked-cell reconstruction inside the model forward, and updates the pretrained
checkpoint with a two-group AdamW (small LR for pretrained weights, larger LR for
the new parameters).

python scripts/meta_finetune.py --steps 300 --out results/meta_ft
"""
from __future__ import annotations

import argparse, json, logging, math, sys, time, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from tabpfn_m import AugmentConfig, TabPFNMConfig, TabPFNMRegressor, upgrade_model
from benchmark_missing import inject_block


# ------------------------------------------------------------ synthetic prior ---
def sample_task(rng: np.random.Generator, n_rows: int, n_feat: int):
    """Random regression task with correlated inputs and a random nonlinear map."""
    z = rng.normal(size=(n_rows, n_feat))
    mix = np.eye(n_feat) + rng.normal(scale=rng.uniform(0.0, 0.8), size=(n_feat, n_feat)) / math.sqrt(n_feat)
    x = z @ mix  # correlated features so missing cells are partly recoverable
    kind = rng.choice(["mlp", "friedman", "linear_inter"])
    if kind == "mlp":
        h = rng.integers(8, 33)
        W1 = rng.normal(size=(n_feat, h)) * rng.uniform(0.5, 2.0)
        act = rng.choice([np.tanh, np.sin, lambda a: np.maximum(a, 0), lambda a: a**2 / 2])
        y = act(x @ W1 + rng.normal(size=h)) @ rng.normal(size=h)
    elif kind == "friedman":
        u = 0.5 + 0.5 * np.tanh(x[:, :5] if n_feat >= 5 else np.pad(x, ((0, 0), (0, 5 - n_feat))))
        y = 10 * np.sin(np.pi * u[:, 0] * u[:, 1]) + 20 * (u[:, 2] - 0.5) ** 2 + 10 * u[:, 3] + 5 * u[:, 4]
    else:
        w = rng.normal(size=n_feat) * (rng.random(n_feat) < 0.7)
        i, j = rng.integers(0, n_feat, size=2)
        y = x @ w + rng.normal() * x[:, i] * x[:, j]
    y = y + rng.normal(scale=rng.uniform(0.05, 0.3) * (y.std() + 1e-6), size=n_rows)
    return x.astype(np.float32), y.astype(np.float32)


def make_batch(rng, batch: int, n_rows: int, n_train: int):
    n_feat = int(rng.integers(4, 11))
    xs, ys = zip(*[sample_task(rng, n_rows, n_feat) for _ in range(batch)])
    x = torch.tensor(np.stack(xs, axis=1))  # (s, B, f)
    y = torch.tensor(np.stack(ys, axis=1))  # (s, B)
    mu, sd = y[:n_train].mean(0, keepdim=True), y[:n_train].std(0, keepdim=True) + 1e-6
    y = (y - mu) / sd
    return x, y


# ------------------------------------------------------------------ eval task ---
def eval_task(seed: int):
    X, y = make_friedman1(n_samples=700, n_features=7, noise=0.5, random_state=seed)
    Xm, _ = inject_block(X, np.random.default_rng(seed + 1), n_sources=4, keep=(3, 4), return_src=True)
    return Xm[:500], y[:500], Xm[500:], y[500:]


def evaluate(state: dict | None, cfg: TabPFNMConfig, seeds=(0, 1)) -> float:
    r2s = []
    for s in seeds:
        Xtr, ytr, Xte, yte = eval_task(s)
        est = TabPFNMRegressor(device="cpu", n_estimators=2, random_state=0, m_config=cfg, m_weights=state)
        est.fit(Xtr, ytr)
        r2s.append(r2_score(yte, est.predict(Xte)))
    return float(np.mean(r2s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--lr-base", type=float, default=1e-5)
    ap.add_argument("--lr-new", type=float, default=5e-3)
    ap.add_argument("--freeze-base", action="store_true")
    ap.add_argument("--recon", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/meta_ft"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

    # Take model and criterion from the estimator so training and evaluation use
    # the same default checkpoint (the raw loader's "v2" file differs from it).
    cfg = TabPFNMConfig(alpha_init=0.0, learn_alpha=True, recon_weight=a.recon, augment=AugmentConfig())
    src_est = TabPFNMRegressor(device="cpu", n_estimators=1, m_config=cfg)
    src_est._initialize_model_variables()
    model = src_est.models_[0]
    criterion = src_est.znorm_space_bardist_
    model.recompute_layer = False  # activation checkpointing not needed at this size
    model.m_training = True
    model._m_generator = torch.Generator().manual_seed(a.seed)
    new = {id(p): p for p in model.m_new_parameters()}
    base = [p for p in model.parameters() if id(p) not in new]
    groups = [{"params": list(new.values()), "lr": a.lr_new, "weight_decay": 0.0}]
    if a.freeze_base:
        for p in base: p.requires_grad_(False)
    else:
        groups.append({"params": base, "lr": a.lr_base, "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    eval_cfg = TabPFNMConfig(alpha_init=0.0, learn_alpha=True)
    log = []
    base_r2 = evaluate(None, TabPFNMConfig.baseline())
    print(f"eval: pretrained TabPFN R2 {base_r2:.4f}", flush=True)
    n_train = int(a.rows * a.train_frac)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = make_batch(rng, a.batch, a.rows, n_train)
        with torch.enable_grad():
            out = model(x, y[:n_train])  # (s_test, B, bars)
            task_loss = criterion(out, y[n_train:]).mean()
            aux = model.m_pop_aux_loss()
            loss = task_loss + (a.recon * aux if aux is not None else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step(); sched.step()
        rec = dict(step=step, loss=float(loss), task=float(task_loss), aux=float(aux) if aux is not None else None,
                   alpha_mean=float(np.mean(model.m_alphas())), absence=float(model.m_absence_embedding.norm()))
        log.append(rec)
        if step % 10 == 0:
            print(f"step {step:4d} loss {rec['loss']:.4f} task {rec['task']:.4f} aux {rec['aux']} "
                  f"alpha {rec['alpha_mean']:+.3f} |abs| {rec['absence']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(state, a.out / "tabpfn_m_meta.pt")
            was = model.m_training; model.m_training = False
            r2 = evaluate(state, eval_cfg)
            model.m_training = was
            rec["eval_r2"] = r2
            print(f"eval @ {step}: TabPFN-M R2 {r2:.4f} (pretrained TabPFN {base_r2:.4f}); alphas {np.round(model.m_alphas(), 2).tolist()}", flush=True)
            (a.out / "log.json").write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, base_r2=base_r2, log=log), indent=1))


if __name__ == "__main__":
    main()
