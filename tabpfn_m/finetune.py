"""Fine-tuning TabPFN-M from the pretrained TabPFN checkpoint.

Builds on tabpfn's ``FinetunedTabPFNRegressor``/``FinetunedTabPFNClassifier``. The
estimator being fine-tuned is a TabPFN-M estimator, so every forward pass applies
block-missingness augmentation and the masked-cell reconstruction objective inside
the model. The reconstruction loss is added to the task loss here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from contextlib import contextmanager

from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor
from tabpfn.finetuning import finetuned_base as _fb

from tabpfn_m.config import AugmentConfig, TabPFNMConfig
from tabpfn_m.estimator import TabPFNMClassifier, TabPFNMRegressor


def default_finetune_config(
    *, recon_weight: float = 0.1, alpha_init: float = 0.0
) -> TabPFNMConfig:
    """Config used while fine-tuning: all three components on."""
    return TabPFNMConfig(
        feature_mask=True,
        absence_embedding=True,
        pattern_bias=True,
        alpha_init=alpha_init,
        learn_alpha=True,
        bias_test_only=True,
        recon_weight=recon_weight,
        recon_mask_frac=0.15,
        augment=AugmentConfig(),
    )


@contextmanager
def _param_group_optimizer(model_getter, new_lr: float, freeze_base: bool):
    """Route tabpfn's optimizer construction through two parameter groups.

    The pretrained weights keep the fine-tuner's learning rate; the new TabPFN-M
    parameters (alpha per layer, absence embedding, reconstruction head) get
    ``new_lr``. AdamW steps scale with the learning rate, not the gradient, so a
    separate group is the only way to let alpha move in a short fine-tune.
    """
    original = _fb.get_and_init_optimizer

    def patched(model_parameters, learning_rate, weight_decay, checkpoint_path=None, device="cpu"):
        model = model_getter()
        new = {id(p): p for p in model.m_new_parameters()}
        base = [p for p in model.parameters() if id(p) not in new]
        if freeze_base:
            for p in base:
                p.requires_grad_(False)
        groups = [{"params": list(new.values()), "lr": new_lr, "weight_decay": 0.0}]
        if not freeze_base:
            groups.insert(0, {"params": base, "lr": learning_rate, "weight_decay": weight_decay})
        return original(groups, learning_rate=learning_rate, weight_decay=weight_decay,
                        checkpoint_path=checkpoint_path, device=device)

    _fb.get_and_init_optimizer = patched
    try:
        yield
    finally:
        _fb.get_and_init_optimizer = original


class _MFinetuneMixin:
    m_config: TabPFNMConfig
    m_freeze_base: bool
    m_new_param_lr: float
    finetuned_estimator_: Any

    def fit(self, X, y, *args, **kwargs):  # type: ignore[override]
        with _param_group_optimizer(self._m_model, self.m_new_param_lr, self.m_freeze_base):
            return super().fit(X, y, *args, **kwargs)  # type: ignore[misc]

    def _m_estimator_cls(self):
        raise NotImplementedError

    def _create_estimator(self, config: dict[str, Any]):  # type: ignore[override]
        est = self._m_estimator_cls()(
            **config,
            fit_mode="batched",
            differentiable_input=False,
            m_config=self.m_config,
        )
        return est

    def _m_model(self):
        return self.finetuned_estimator_.models_[0]

    def _forward_with_loss(self, batch):  # type: ignore[override]
        model = self._m_model()
        model.m_training = True
        model._m_aux_losses = []
        loss = super()._forward_with_loss(batch)  # type: ignore[misc]
        aux = model.m_pop_aux_loss()
        if aux is not None and self.m_config.recon_weight > 0:
            loss = loss + self.m_config.recon_weight * aux
        return loss

    def m_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights of the fine-tuned model (full state dict, CPU)."""
        return {k: v.detach().cpu() for k, v in self._m_model().state_dict().items()}

    def m_save(self, path: str | Path) -> None:
        torch.save(self.m_state_dict(), path)

    def m_alphas(self) -> list[float]:
        return self._m_model().m_alphas()


class FinetunedTabPFNMRegressor(_MFinetuneMixin, FinetunedTabPFNRegressor):
    """Fine-tune a TabPFN-M regressor.

    Extra parameters:
        m_config: TabPFN-M configuration used during training (defaults to all
            components on with ``recon_weight=0.1``).
        m_freeze_base: train only the new TabPFN-M parameters (alpha per layer,
            absence embedding, reconstruction head) and leave the pretrained weights
            untouched.
        m_new_param_lr: learning rate of the new parameters (the pretrained weights
            use ``learning_rate``).
    """

    def __init__(
        self,
        *args,
        m_config: TabPFNMConfig | None = None,
        m_freeze_base: bool = False,
        m_new_param_lr: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.m_config = m_config if m_config is not None else default_finetune_config()
        self.m_freeze_base = m_freeze_base
        self.m_new_param_lr = m_new_param_lr

    def _m_estimator_cls(self):
        return TabPFNMRegressor


class FinetunedTabPFNMClassifier(_MFinetuneMixin, FinetunedTabPFNClassifier):
    """Fine-tune a TabPFN-M classifier."""

    def __init__(
        self,
        *args,
        m_config: TabPFNMConfig | None = None,
        m_freeze_base: bool = False,
        m_new_param_lr: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.m_config = m_config if m_config is not None else default_finetune_config()
        self.m_freeze_base = m_freeze_base
        self.m_new_param_lr = m_new_param_lr

    def _m_estimator_cls(self):
        return TabPFNMClassifier
