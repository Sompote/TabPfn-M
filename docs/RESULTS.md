# TabPFN-M experiment results

All runs on CPU with the TabPFN v2.5 default checkpoint, 6 Sep 2026. Raw CSVs are in `results/`.

## Synthetic and small public datasets

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


## Compaction database (partial, run stopped 6 Sep 2026)

2,854 records from 6 public sources, 7 numeric features, target MDD (Mg/m³), real missingness
(one figshare source never reports sand_pct; LL/PL 7 % missing in LTPP). Zero-shot, 4 ensemble
members. R² on the held-out rows. OMC target, source-block design and grouped fold 5 were not run.

| folds | held-out | n test | HGB | TabPFN | TabPFN-M mask | TabPFN-M mask + bias |
|---|---|---|---|---|---|---|
| group5 | fold0 | 571 | 0.739 | 0.758 | 0.758 | 0.757 |
| group5 | fold1 | 571 | 0.743 | 0.750 | 0.751 | 0.750 |
| group5 | fold2 | 571 | 0.554 | 0.598 | 0.598 | 0.597 |
| group5 | fold3 | 571 | 0.748 | 0.722 | 0.722 | 0.723 |
| group5 | fold4 | 570 | 0.726 | 0.736 |  |  |
| loso | figshare:28681187 | 395 | 0.417 | 0.496 | 0.492 | 0.492 |
| loso | figshare:32955851 | 169 | 0.196 | 0.227 | 0.226 | 0.223 |
| loso | ltpp-sdr39 | 1976 | 0.608 | 0.381 | 0.380 | 0.393 |
| loso | zenodo:14251190 | 38 | 0.185 | 0.245 | 0.240 | 0.237 |
| loso | zenodo:19242689 | 7 | -0.023 | -0.425 | -0.422 | -0.432 |
| loso | zenodo:20737270 | 269 | 0.655 | 0.665 | 0.665 | 0.666 |
| group5 | **mean** (7-row source excluded) | | 0.702 | 0.713 | 0.707 | 0.707 |
| loso | **mean** (7-row source excluded) | | 0.412 | 0.403 | 0.401 | 0.402 |

Fine-tuned on the other five sources (10 epochs, LR 1e-5, new-parameter LR 1e-2), LOSO:

| held-out | n test | TabPFN fine-tuned | TabPFN-M fine-tuned | TabPFN zero-shot |
|---|---|---|---|---|
| figshare:28681187 | 395 | 0.486 | 0.488 | 0.496 |
| figshare:32955851 | 169 | 0.205 |  | 0.227 |

TabPFN-M tracks TabPFN within 0.012 R² on every fold. Fine-tuning on the other sources lowers
transfer on both folds completed. HGB beats TabPFN by 0.23 R² when the large LTPP source is held out.
