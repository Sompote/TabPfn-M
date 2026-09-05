"""Configuration dataclasses for TabPFN-M."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AugmentConfig:
    """Training-time block-missingness augmentation.

    Rows of each in-context dataset are split into ``n_sources`` pseudo-sources.
    Every pseudo-source observes a fixed random subset of the features and hides
    the rest, which mimics data merged from laboratories running different test
    programmes. A small MCAR sprinkle is added on top.
    """

    p_apply: float = 0.8
    """Probability that a dataset in the batch is augmented at all."""
    n_sources_min: int = 2
    n_sources_max: int = 6
    keep_frac_min: float = 0.3
    """Minimum fraction of features a pseudo-source observes."""
    keep_frac_max: float = 0.9
    p_mcar: float = 0.03
    """Cell-wise MCAR probability added on top of the block structure."""
    seed: int | None = None


@dataclass
class TabPFNMConfig:
    """Switches for the three TabPFN-M components.

    With every switch off and ``alpha_init=0`` the model is numerically identical
    to the pretrained TabPFN checkpoint, which is what the ablation relies on.
    """

    feature_mask: bool = True
    """Component 1a: mask fully missing feature groups as keys in feature attention."""
    absence_embedding: bool = True
    """Component 1b: add a learned vector to the tokens of fully missing groups."""
    pattern_bias: bool = True
    """Component 2: additive Jaccard-overlap bias in the item (row) attention."""
    alpha_init: float = 0.0
    """Initial per-layer bias scale. 0 recovers the pretrained model exactly."""
    learn_alpha: bool = True
    """Whether alpha is a trainable parameter (fine-tuning) or fixed (zero-shot)."""
    bias_test_only: bool = True
    """Apply the pattern bias only where test rows query the training rows, leaving
    the train-train attention (the in-context representation) untouched."""
    recon_weight: float = 0.0
    """Component 3: weight of the masked-cell reconstruction loss (training only)."""
    recon_mask_frac: float = 0.15
    """Fraction of observed cells hidden for the reconstruction objective."""
    augment: AugmentConfig | None = None
    """Component 3: block-missingness augmentation (training only)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def baseline(cls) -> TabPFNMConfig:
        """Plain TabPFN: every component off."""
        return cls(
            feature_mask=False,
            absence_embedding=False,
            pattern_bias=False,
            alpha_init=0.0,
            learn_alpha=False,
            recon_weight=0.0,
            augment=None,
        )
