# Missingness benchmark

seeds: [0, 1, 2], n_estimators: 4, max_rows: 600

## compaction_MDD / block2-5

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_mask | 0.5640 | 0.0232 | 0.1460 |
| tabpfn | 0.5636 | 0.0229 | 0.1461 |
| m_mask_bias | 0.5600 | 0.0271 | 0.1466 |
| m_bias | 0.5595 | 0.0268 | 0.1467 |
| tabpfn_mean | 0.5574 | 0.0301 | 0.1470 |
| hgb | 0.5233 | 0.0495 | 0.1524 |

## compaction_MDD / block3-4of7

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_mask | 0.5366 | 0.0418 | 0.1503 |
| tabpfn | 0.5366 | 0.0419 | 0.1503 |
| tabpfn_mean | 0.5344 | 0.0287 | 0.1508 |
| m_bias | 0.5320 | 0.0381 | 0.1511 |
| m_mask_bias | 0.5317 | 0.0385 | 0.1512 |
| hgb | 0.4806 | 0.0260 | 0.1594 |

## compaction_MDD / mcar30

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.6608 | 0.0421 | 0.1288 |
| m_mask | 0.6605 | 0.0428 | 0.1288 |
| m_bias | 0.6567 | 0.0392 | 0.1296 |
| m_mask_bias | 0.6563 | 0.0397 | 0.1296 |
| tabpfn_mean | 0.6382 | 0.0493 | 0.1330 |
| hgb | 0.5863 | 0.0334 | 0.1423 |

## compaction_MDD / mcar50

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.5329 | 0.0415 | 0.1512 |
| m_mask | 0.5325 | 0.0404 | 0.1513 |
| m_bias | 0.5295 | 0.0416 | 0.1518 |
| m_mask_bias | 0.5293 | 0.0393 | 0.1518 |
| tabpfn_mean | 0.4991 | 0.0393 | 0.1566 |
| hgb | 0.4550 | 0.0453 | 0.1633 |

## compaction_MDD / none

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_bias | 0.7716 | 0.0296 | 0.1057 |
| m_mask_bias | 0.7716 | 0.0296 | 0.1057 |
| tabpfn | 0.7716 | 0.0296 | 0.1057 |
| tabpfn_mean | 0.7716 | 0.0296 | 0.1057 |
| m_mask | 0.7716 | 0.0296 | 0.1057 |
| hgb | 0.7412 | 0.0515 | 0.1125 |

## diabetes / block2-5

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.1687 | 0.1145 | 66.2908 |
| m_mask | 0.1678 | 0.1130 | 66.3334 |
| m_bias | 0.1665 | 0.1139 | 66.3784 |
| m_mask_bias | 0.1658 | 0.1120 | 66.4132 |
| tabpfn_mean | 0.1568 | 0.1038 | 66.7971 |
| hgb | -0.0382 | 0.1802 | 73.9626 |

## diabetes / block3-4of7

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn_mean | 0.2387 | 0.0170 | 63.6408 |
| m_mask | 0.2377 | 0.0211 | 63.6845 |
| tabpfn | 0.2365 | 0.0225 | 63.7306 |
| m_mask_bias | 0.2297 | 0.0254 | 64.0084 |
| m_bias | 0.2274 | 0.0248 | 64.1021 |
| hgb | 0.0743 | 0.1085 | 70.0774 |

## diabetes / mcar30

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_mask | 0.3191 | 0.0723 | 60.0678 |
| m_mask_bias | 0.3153 | 0.0713 | 60.2381 |
| tabpfn | 0.3149 | 0.0730 | 60.2516 |
| m_bias | 0.3110 | 0.0723 | 60.4268 |
| tabpfn_mean | 0.2971 | 0.0738 | 61.0300 |
| hgb | 0.2109 | 0.1177 | 64.6043 |

## diabetes / mcar50

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn_mean | 0.3293 | 0.0390 | 59.7099 |
| tabpfn | 0.3240 | 0.0343 | 59.9501 |
| m_bias | 0.3234 | 0.0314 | 59.9778 |
| m_mask | 0.3211 | 0.0382 | 60.0698 |
| m_mask_bias | 0.3208 | 0.0356 | 60.0868 |
| hgb | 0.2365 | 0.0395 | 63.7106 |

## diabetes / none

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_mask | 0.4239 | 0.0557 | 55.2618 |
| m_bias | 0.4239 | 0.0557 | 55.2618 |
| m_mask_bias | 0.4239 | 0.0557 | 55.2618 |
| tabpfn | 0.4239 | 0.0557 | 55.2618 |
| tabpfn_mean | 0.4239 | 0.0557 | 55.2618 |
| hgb | 0.3102 | 0.0420 | 60.5299 |

## friedman1_7f / block2-5

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| m_mask | 0.4178 | 0.1361 | 3.8481 |
| tabpfn_mean | 0.4177 | 0.1314 | 3.8486 |
| tabpfn | 0.4172 | 0.1341 | 3.8506 |
| m_mask_bias | 0.4154 | 0.1382 | 3.8556 |
| m_bias | 0.4150 | 0.1355 | 3.8578 |
| hgb | 0.3040 | 0.1775 | 4.2003 |

## friedman1_7f / block3-4of7

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.4121 | 0.0493 | 3.8879 |
| m_mask | 0.4117 | 0.0495 | 3.8891 |
| m_bias | 0.4087 | 0.0536 | 3.8983 |
| m_mask_bias | 0.4084 | 0.0532 | 3.8992 |
| tabpfn_mean | 0.4080 | 0.0468 | 3.9009 |
| hgb | 0.3053 | 0.0872 | 4.2251 |

## friedman1_7f / mcar30

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.6662 | 0.0180 | 2.9294 |
| m_bias | 0.6655 | 0.0121 | 2.9331 |
| m_mask | 0.6649 | 0.0203 | 2.9351 |
| m_mask_bias | 0.6639 | 0.0146 | 2.9397 |
| tabpfn_mean | 0.6303 | 0.0376 | 3.0822 |
| hgb | 0.5607 | 0.0210 | 3.3635 |

## friedman1_7f / mcar50

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.4973 | 0.0813 | 3.5830 |
| m_mask | 0.4967 | 0.0803 | 3.5854 |
| m_bias | 0.4949 | 0.0857 | 3.5900 |
| m_mask_bias | 0.4944 | 0.0856 | 3.5922 |
| tabpfn_mean | 0.4563 | 0.1017 | 3.7218 |
| hgb | 0.4017 | 0.1097 | 3.9054 |

## friedman1_7f / none

| method | R2 mean | R2 std | RMSE mean |
|---|---|---|---|
| tabpfn | 0.9880 | 0.0008 | 0.5558 |
| tabpfn_mean | 0.9880 | 0.0008 | 0.5558 |
| m_bias | 0.9880 | 0.0008 | 0.5558 |
| m_mask | 0.9880 | 0.0008 | 0.5558 |
| m_mask_bias | 0.9880 | 0.0008 | 0.5558 |
| hgb | 0.9227 | 0.0131 | 1.4041 |
