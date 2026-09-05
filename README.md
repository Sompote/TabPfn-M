# TabPFN-M: TabPFN for tables with incomplete feature sets

TabPFN-M adapts a pretrained TabPFN v2 checkpoint to data where different rows
carry different feature subsets. The target case is a merged database: seven
features in the schema, but one laboratory reports three of them and another
four. Missingness is then block-structured by source, not random per cell.

Plain TabPFN handles NaN by mean-imputing the cell and appending a NaN indicator
to the token. TabPFN-M keeps that encoder and changes how attention treats the
missing tokens. All changes are additive, so with every switch off the model
equals the pretrained TabPFN bit for bit (tested).

## The three components

1. **Observed-only feature attention.** In the attention between the feature
   tokens of one row, tokens of fully missing feature groups are masked as keys.
   A row's representation is built only from what was measured. A learned
   absence vector (zero-initialised) marks the missing group tokens.
   `tabpfn_m/attention.py::MissingAwareFeatureAttention`
2. **Pattern-overlap attention bias.** In the attention between rows, each test
   row receives an additive score `alpha_l * Jaccard(m_i, m_j)` towards training
   rows whose observed-feature set overlaps its own. `alpha_l` is one scalar per
   layer, initialised at 0. Thinking rows get the query's mean overlap so the
   bias is neutral for them; with complete data the bias is uniform and the
   softmax is unchanged. By default the bias is applied only where test rows
   query training rows, which keeps the in-context representation intact.
   `tabpfn_m/attention.py::PatternBiasedItemAttention`
3. **Block-missingness fine-tuning with masked-cell reconstruction.** The prior
   used to pretrain TabPFN is not public, so the components are trained by
   fine-tuning from the released checkpoint. During training each in-context
   dataset is split into pseudo-sources that observe fixed random feature
   subsets, and a fraction of observed cells is hidden and reconstructed from
   the final feature tokens by a linear head (smooth-L1 on standardised
   values). `tabpfn_m/augment.py`, `tabpfn_m/finetune.py`

## Prior art to cite as such

- TabPFN v2 (Hollmann et al., 2025): cell-wise missingness in the prior and
  the NaN indicator. This is the baseline, not a contribution.
- NAIM (Caruso et al., 2024): attention masks over missing features in a
  transformer trained per dataset. Component 1 brings the idea to in-context
  learning with a pretrained checkpoint.
- ReMasker / VIME: masked reconstruction as a self-supervised objective for
  tabular data. Component 3 uses it as an auxiliary loss during fine-tuning.
- The pattern-overlap bias (component 2) is, to my knowledge, new. Check the
  literature before claiming it.

## What the code does and does not do

- Runs on the installed `tabpfn` 6.4.1 (TabPFN v2 architecture, open weights).
  The `tabpfn/` folder is an upstream clone kept for reference; it needs torch
  2.5+, which has no wheels for Intel macOS.
- No KV cache and no `save_peak_mem_factor` chunking inside the wrapped
  attention. Fine for datasets up to a few thousand rows on CPU.
- Regressor and classifier estimators exist; only the regressor has been
  exercised so far.

## Use

```python
from tabpfn_m import TabPFNMRegressor, TabPFNMConfig

# zero-shot: observed-only attention + pattern bias with a fixed alpha
est = TabPFNMRegressor(device="cpu", m_config=TabPFNMConfig(alpha_init=1.0, learn_alpha=False))
est.fit(X_train_with_nans, y_train)
pred = est.predict(X_test_with_nans)

# fine-tune on your own data, then reuse the weights
from tabpfn_m.finetune import FinetunedTabPFNMRegressor
ft = FinetunedTabPFNMRegressor(device="cpu", epochs=30, learning_rate=1e-5, m_new_param_lr=1e-2)
ft.fit(X_train_with_nans, y_train)
ft.m_save("tabpfn_m.pt")
est = TabPFNMRegressor(device="cpu", m_weights="tabpfn_m.pt")
```

Tests: `python -m pytest tests -q`. Benchmark: `python scripts/benchmark_missing.py`.
Fine-tuning demo: `python scripts/finetune_demo.py`.

## Results so far (6 Sep 2026, TabPFN v2.5 default checkpoint, CPU)

Zero-shot, no training (`results/zero_shot`, 3 seeds, 4 ensemble members,
R² mean over seeds):

| dataset / design | HGB | TabPFN mean-impute | TabPFN | M mask | M mask+bias(1) |
|---|---|---|---|---|---|
| friedman1 / block 3-4 of 7 | 0.305 | 0.408 | 0.412 | 0.412 | 0.408 |
| friedman1 / mcar50 | 0.402 | 0.456 | 0.497 | 0.497 | 0.494 |
| compaction MDD / block 3-4 of 7 | 0.481 | 0.534 | 0.537 | 0.537 | 0.532 |
| compaction MDD / mcar50 | 0.455 | 0.499 | 0.533 | 0.532 | 0.529 |
| diabetes / block 3-4 of 7 | 0.074 | 0.239 | 0.237 | 0.238 | 0.230 |

Mean R² difference to TabPFN over all 15 cells: 0.000 (mask), -0.003 (mask +
bias). Untrained, the components are inert. TabPFN's NaN indicator already
carries the pattern information: with a source-dependent target offset the
pattern reveals the source, and TabPFN on block-missing data reaches R² 0.83
where the same model on complete data (source invisible) reaches 0.17.

Single-dataset fine-tuning (`results/finetune_friedman*`, 500 rows, 30 epochs)
moves the per-layer alphas by at most 0.1 and changes R² by less than 0.01
against a plain fine-tuned TabPFN control. The new parameters are prior-level
and need meta-fine-tuning across many tasks (`scripts/meta_finetune.py`).

Meta-fine-tuning across random synthetic block-missing tasks
(`results/meta_ft*`, 300 steps, 4 tasks of 300 rows per step, held-out Friedman
block task, 2 seeds):

| step | full fine-tune (base LR 1e-5, new LR 5e-3) | frozen base (new params only) |
|---|---|---|
| 0 | 0.349 | 0.349 |
| 100 | 0.340 | 0.348 |
| 200 | 0.335 | 0.348 |
| 300 | 0.335 | 0.348 |

The learned alphas settle in a stable pattern (about -0.2 in layers 7 to 10,
+0.3 in layer 16) that is the same in both runs, so the gradient signal is
consistent, but the held-out score does not move. Updating the pretrained
weights on my synthetic prior costs 0.014 R². Conclusion at this stage: on top
of TabPFN v2.5, whose NaN indicator already encodes the pattern, the three
components give no measurable gain in this setting. A different data regime
(real multi-source data with source effects, MNAR) or a richer prior for
meta-fine-tuning is needed before any claim can be made.

## Evaluation plan

Ablate the three components separately (the config switches exist for this),
on synthetic block missingness and on the compaction database with
leave-one-source-out folds, against TabPFN with NaN, TabPFN with mean
imputation, and HistGradientBoosting. Results live in `results/`.
