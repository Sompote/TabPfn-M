"""TabPFN-M: TabPFN with missingness-aware attention for incomplete feature sets.

Three separable components on top of a pretrained TabPFN v2 checkpoint:

1. Observed-only feature attention: a row's tokens attend only to feature groups
   that are observed for that row (fully missing groups are masked as keys) and a
   learned absence vector marks fully missing groups.
2. Pattern-overlap attention bias: in the row-wise (item) attention every query row
   receives an additive bias ``alpha_l * Jaccard(m_i, m_j)`` towards training rows
   whose observed-feature set overlaps its own.
3. Block-missingness fine-tuning with masked-cell reconstruction: the pretrained
   weights are adapted with source-structured missingness augmentation and an
   auxiliary loss that reconstructs artificially hidden cells.
"""

from tabpfn_m.config import TabPFNMConfig, AugmentConfig
from tabpfn_m.model import TabPFNMTransformer, upgrade_model
from tabpfn_m.estimator import TabPFNMRegressor, TabPFNMClassifier

__all__ = [
    "TabPFNMConfig",
    "AugmentConfig",
    "TabPFNMTransformer",
    "upgrade_model",
    "TabPFNMRegressor",
    "TabPFNMClassifier",
]
