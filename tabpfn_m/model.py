"""TabPFN-M architecture: a pretrained PerFeatureTransformer with missingness-aware
attention, a learned absence embedding, and a masked-cell reconstruction head."""

from __future__ import annotations

from typing import Any

import einops
import torch
from torch import nn

from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.encoders.pipeline_interfaces import TorchPreprocessingPipeline

from tabpfn_m.attention import MissingAwareFeatureAttention, PatternBiasedItemAttention
from tabpfn_m.augment import block_missingness, reconstruction_mask, standardise_targets
from tabpfn_m.config import TabPFNMConfig
from tabpfn_m.context import ContextHolder, build_context


class TabPFNMTransformer(PerFeatureTransformer):
    """PerFeatureTransformer with the three TabPFN-M components.

    Instances are created by :func:`upgrade_model` from a loaded TabPFN model, so the
    pretrained weights are reused and only the new parameters (absence embedding,
    per-layer alpha, reconstruction head) are initialised here.
    """

    m_config: TabPFNMConfig
    m_holder: ContextHolder

    # ------------------------------------------------------------------ setup ---
    def _init_m(self, cfg: TabPFNMConfig) -> None:
        self.m_config = cfg
        holder = ContextHolder()
        holder.feature_mask = cfg.feature_mask
        holder.pattern_bias = cfg.pattern_bias
        holder.bias_test_only = cfg.bias_test_only
        object.__setattr__(self, "m_holder", holder)
        # Infer the embedding size from the thinking tokens or the decoder.
        if self.add_thinking_tokens is not None:
            emsize = self.add_thinking_tokens.row_token_values.shape[-1]
        else:
            emsize = next(self.decoder_dict["standard"].parameters()).shape[-1]
        self.m_absence_embedding = nn.Parameter(torch.zeros(emsize))
        self.m_recon_head = nn.Linear(emsize, self.features_per_group)
        nn.init.zeros_(self.m_recon_head.weight)
        nn.init.zeros_(self.m_recon_head.bias)
        for layer in self.transformer_encoder.layers:
            if layer.self_attn_between_features is not None and not isinstance(
                layer.self_attn_between_features, MissingAwareFeatureAttention
            ):
                layer.self_attn_between_features = MissingAwareFeatureAttention(
                    layer.self_attn_between_features, holder
                )
            if not isinstance(layer.self_attn_between_items, PatternBiasedItemAttention):
                layer.self_attn_between_items = PatternBiasedItemAttention(
                    layer.self_attn_between_items,
                    holder,
                    alpha_init=cfg.alpha_init,
                    learn_alpha=cfg.learn_alpha,
                )
        self._m_aux_losses: list[torch.Tensor] = []
        self._m_generator: torch.Generator | None = None
        self.m_training: bool = False

    def m_new_parameters(self) -> list[nn.Parameter]:
        """Parameters that do not exist in the pretrained checkpoint."""
        params = [self.m_absence_embedding, *self.m_recon_head.parameters()]
        for layer in self.transformer_encoder.layers:
            a = layer.self_attn_between_items
            if isinstance(a, PatternBiasedItemAttention) and isinstance(a.alpha, nn.Parameter):
                params.append(a.alpha)
        return params

    def m_alphas(self) -> list[float]:
        return [
            float(layer.self_attn_between_items.alpha)
            for layer in self.transformer_encoder.layers
            if isinstance(layer.self_attn_between_items, PatternBiasedItemAttention)
        ]

    def m_pop_aux_loss(self) -> torch.Tensor | None:
        """Sum of reconstruction losses accumulated since the last call."""
        if not self._m_aux_losses:
            return None
        loss = torch.stack(self._m_aux_losses).sum()
        self._m_aux_losses = []
        return loss

    def m_set_active(self, *, feature_mask: bool | None = None, pattern_bias: bool | None = None) -> None:
        if feature_mask is not None:
            self.m_holder.feature_mask = feature_mask
        if pattern_bias is not None:
            self.m_holder.pattern_bias = pattern_bias

    # ---------------------------------------------------------------- forward ---
    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        style: torch.Tensor | None = None,
        data_dags: Any = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        assert style is None and data_dags is None
        assert not self.cache_trainset_representation, (
            "TabPFN-M does not support cached train-set representations."
        )
        cfg = self.m_config

        x = {"main": x} if not isinstance(x, dict) else dict(x)
        y = {"main": y} if not isinstance(y, dict) else dict(y)
        seq_len, batch_size, num_features = x["main"].shape
        if y["main"] is None:
            raise ValueError("TabPFN-M needs training labels in every forward pass.")
        single_eval_pos = int(y["main"].shape[0])

        # -- Component 3 (training only): block-missingness augmentation and the
        #    reconstruction targets, applied to the raw NaN-carrying input.
        recon_targets = recon_cells = None
        # Training-time behaviour is gated on an explicit flag set by the fine-tuner
        # and on autograd being enabled: tabpfn never toggles train()/eval() and
        # predicts under inference mode.
        if getattr(self, "m_training", False) and torch.is_grad_enabled():
            if cfg.augment is not None:
                x["main"], _ = block_missingness(x["main"], cfg.augment, self._m_generator)
            if cfg.recon_weight > 0:
                recon_cells = reconstruction_mask(
                    x["main"], cfg.recon_mask_frac, self._m_generator
                )
                recon_targets = standardise_targets(x["main"], single_eval_pos)
                x_masked = x["main"].clone()
                x_masked[recon_cells] = float("nan")
                x["main"] = x_masked

        # -- Missingness context from the raw input, before padding and grouping.
        n_think = self.add_thinking_tokens.num_thinking_rows if self.add_thinking_tokens else 0
        use_ctx = cfg.feature_mask or cfg.pattern_bias or cfg.absence_embedding
        ctx = (
            build_context(
                x["main"],
                features_per_group=self.features_per_group,
                num_thinking_rows=n_think,
                num_train_rows=single_eval_pos,
            )
            if use_ctx
            else None
        )
        self.m_holder.ctx = ctx if (cfg.feature_mask or cfg.pattern_bias) else None

        try:
            return self._forward_body(
                x,
                y,
                ctx=ctx,
                seq_len=seq_len,
                batch_size=batch_size,
                num_features=num_features,
                single_eval_pos=single_eval_pos,
                only_return_standard_out=only_return_standard_out,
                categorical_inds=categorical_inds,
                force_recompute_layer=force_recompute_layer,
                save_peak_memory_factor=save_peak_memory_factor,
                recon_targets=recon_targets,
                recon_cells=recon_cells,
            )
        finally:
            self.m_holder.ctx = None

    def _forward_body(  # noqa: C901, PLR0912, PLR0913
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        *,
        ctx,
        seq_len: int,
        batch_size: int,
        num_features: int,
        single_eval_pos: int,
        only_return_standard_out: bool,
        categorical_inds: list[list[int]] | None,
        force_recompute_layer: bool,
        save_peak_memory_factor: int | None,
        recon_targets: torch.Tensor | None,
        recon_cells: torch.Tensor | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        cfg = self.m_config
        n = self.features_per_group

        # Pad features to a multiple of features_per_group and group them.
        for k in x:
            missing_to_next = (n - (x[k].shape[2] % n)) % n
            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len, batch_size, missing_to_next,
                            device=x[k].device, dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )
        for k in x:
            x[k] = einops.rearrange(x[k], "s b (f n) -> b s f n", n=n)

        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            n_subgroups = x["main"].shape[2]
            categorical_inds_to_use = []
            for batch_idx in range(batch_size):
                for subgroup in range(n_subgroups):
                    lo, hi = subgroup * n, (subgroup + 1) * n
                    categorical_inds_to_use.append(
                        [i - lo for i in categorical_inds[batch_idx] if lo <= i < hi]
                    )

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)
            y[k] = y[k].transpose(0, 1)
            if y[k].shape[1] < x["main"].shape[1]:
                assert y[k].shape[1] == single_eval_pos
                y[k] = torch.cat(
                    (
                        y[k],
                        torch.nan * torch.zeros(
                            y[k].shape[0], x["main"].shape[1] - y[k].shape[1], y[k].shape[2],
                            device=y[k].device, dtype=y[k].dtype,
                        ),
                    ),
                    dim=1,
                )
            y[k] = y[k].transpose(0, 1)
        y["main"][single_eval_pos:] = torch.nan

        embedded_y = self.y_encoder(
            y, single_eval_pos=single_eval_pos, cache_trainset_representation=False
        ).transpose(0, 1)
        del y

        extra_encoders_args: dict[str, Any] = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder, TorchPreprocessingPipeline
        ):
            extra_encoders_args["categorical_inds"] = sum(categorical_inds_to_use, [])

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.encoder(
                x, single_eval_pos=single_eval_pos, cache_trainset_representation=False,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x, embedded_y, data_dags=None, num_features=num_features,
            seq_len=seq_len, cache_embeddings=False, use_cached_embeddings=False,
        )

        # -- Component 1b: learned absence vector on fully missing group tokens.
        if cfg.absence_embedding and ctx is not None:
            n_think = ctx.num_thinking_rows
            missing_BSG = ctx.group_missing_BRG[:, n_think:]
            embedded_x = embedded_x + missing_BSG[..., None].to(embedded_x.dtype) * self.m_absence_embedding

        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        if torch.isnan(embedded_input).any():
            raise ValueError("NaNs in the encoded input; the NaN-handling encoder failed.")
        del embedded_y, embedded_x

        if self.add_thinking_tokens is not None:
            embedded_input, single_eval_pos = self.add_thinking_tokens(
                embedded_input, single_eval_pos
            )

        encoder_out = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            recompute_layer=self.recompute_layer or force_recompute_layer,
            save_peak_mem_factor=save_peak_memory_factor,
        )
        del embedded_input

        # -- Component 3: masked-cell reconstruction from the final feature tokens.
        if recon_targets is not None and recon_cells is not None and recon_cells.any():
            n_think = ctx.num_thinking_rows if ctx is not None else 0
            feat_tokens = encoder_out[:, n_think:, :-1]  # (b, s, G, e)
            pred_BSGn = self.m_recon_head(feat_tokens)
            pred_SBF = einops.rearrange(pred_BSGn, "b s g n -> s b (g n)")[..., :num_features]
            diff = pred_SBF[recon_cells] - recon_targets[recon_cells].to(pred_SBF.dtype)
            self._m_aux_losses.append(torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff)))

        test_encoder_out = encoder_out[:, single_eval_pos:, -1].transpose(0, 1)
        if only_return_standard_out:
            assert self.decoder_dict is not None
            return self.decoder_dict["standard"](test_encoder_out)

        output_decoded = (
            {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
            if self.decoder_dict is not None
            else {}
        )
        thinking_rows_offset = (
            self.add_thinking_tokens.num_thinking_rows if self.add_thinking_tokens is not None else 0
        )
        output_decoded["train_embeddings"] = encoder_out[
            :, thinking_rows_offset:single_eval_pos, -1
        ].transpose(0, 1)
        output_decoded["test_embeddings"] = test_encoder_out
        return output_decoded


def upgrade_model(model: PerFeatureTransformer, cfg: TabPFNMConfig) -> TabPFNMTransformer:
    """Turn a loaded TabPFN model into a TabPFN-M model in place (weights shared)."""
    if isinstance(model, TabPFNMTransformer):
        model.m_config = cfg
        model.m_holder.feature_mask = cfg.feature_mask
        model.m_holder.pattern_bias = cfg.pattern_bias
        model.m_holder.bias_test_only = cfg.bias_test_only
        for layer in model.transformer_encoder.layers:
            a = layer.self_attn_between_items
            if isinstance(a, PatternBiasedItemAttention):
                if isinstance(a.alpha, nn.Parameter) != cfg.learn_alpha:
                    val = torch.tensor(float(cfg.alpha_init))
                    if isinstance(a.alpha, nn.Parameter):
                        del a.alpha
                        a.register_buffer("alpha", val)
                    else:
                        del a.alpha
                        a.alpha = nn.Parameter(val)
                else:
                    with torch.no_grad():
                        a.alpha.fill_(float(cfg.alpha_init))
        return model
    assert isinstance(model, PerFeatureTransformer), type(model)
    model.__class__ = TabPFNMTransformer
    model._init_m(cfg)  # type: ignore[attr-defined]
    return model  # type: ignore[return-value]
