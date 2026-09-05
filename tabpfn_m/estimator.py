"""sklearn-style estimators: TabPFN regressor/classifier with TabPFN-M upgrades."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor

from tabpfn_m.config import TabPFNMConfig
from tabpfn_m.model import TabPFNMTransformer, upgrade_model


def _merged_signature(parent_init, extra: dict[str, Any]) -> inspect.Signature:
    sig = inspect.signature(parent_init)
    params = [p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    for name, default in extra.items():
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default))
    return sig.replace(parameters=params)


class _MMixin:
    """Shared upgrade logic. ``m_config`` and ``m_weights`` are sklearn params."""

    m_config: TabPFNMConfig | None
    m_weights: str | Path | dict | None

    def _upgrade_models(self) -> None:
        cfg = self.m_config if self.m_config is not None else TabPFNMConfig()
        for i, model in enumerate(self.models_):  # type: ignore[attr-defined]
            self.models_[i] = upgrade_model(model, cfg)  # type: ignore[attr-defined]
        if self.m_weights is not None:
            state = self.m_weights
            if not isinstance(state, dict):
                state = torch.load(state, map_location="cpu")
            for model in self.models_:  # type: ignore[attr-defined]
                missing, unexpected = model.load_state_dict(state, strict=False)
                if unexpected:
                    raise RuntimeError(f"Unexpected keys in TabPFN-M weights: {unexpected[:5]}")

    def _initialize_model_variables(self):  # type: ignore[override]
        out = super()._initialize_model_variables()  # type: ignore[misc]
        self._upgrade_models()
        return out

    @property
    def m_model(self) -> TabPFNMTransformer:
        return self.models_[0]  # type: ignore[attr-defined]


class TabPFNMRegressor(_MMixin, TabPFNRegressor):
    """TabPFNRegressor whose loaded model is upgraded to TabPFN-M.

    Extra parameters:
        m_config: component switches (default: feature mask, absence embedding and
            pattern bias on, ``alpha_init=0`` so the untuned model matches TabPFN
            unless alpha is set or fine-tuned weights are loaded).
        m_weights: path or state dict with fine-tuned TabPFN-M weights.
    """

    def __init__(self, *args, m_config: TabPFNMConfig | None = None, m_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.m_config = m_config
        self.m_weights = m_weights

    __init__.__signature__ = _merged_signature(  # type: ignore[attr-defined]
        TabPFNRegressor.__init__, {"m_config": None, "m_weights": None}
    )


class TabPFNMClassifier(_MMixin, TabPFNClassifier):
    """TabPFNClassifier whose loaded model is upgraded to TabPFN-M."""

    def __init__(self, *args, m_config: TabPFNMConfig | None = None, m_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.m_config = m_config
        self.m_weights = m_weights

    __init__.__signature__ = _merged_signature(  # type: ignore[attr-defined]
        TabPFNClassifier.__init__, {"m_config": None, "m_weights": None}
    )
