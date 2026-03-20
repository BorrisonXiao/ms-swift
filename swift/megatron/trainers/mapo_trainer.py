# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from megatron.core import mpu
from megatron.training import get_args, get_model, get_wandb_writer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model

from swift.megatron.utils import forward_step_helper
from swift.model import get_model_info_meta
from swift.rlhf_trainers.utils import pad_logps_back_to_batch
from swift.utils import get_logger
from .grpo_trainer import MegatronGRPOTrainer
from .utils import gather, gather_object
from .vocab_parallel_utils import compute_logps_and_entropy_from_logits

logger = get_logger()


class MegatronMAPOTrainer(MegatronGRPOTrainer):

    @staticmethod
    def _validate_text_ref_compatibility(args, text_ref_info):
        if text_ref_info.model_type != args.model_type:
            raise ValueError(
                f'text_ref_model model_type ({text_ref_info.model_type}) must match current model_type '
                f'({args.model_type})')

        policy_vocab_size = getattr(getattr(args.model_info, 'config', None), 'vocab_size', None)
        text_ref_vocab_size = getattr(getattr(text_ref_info, 'config', None), 'vocab_size', None)
        if policy_vocab_size is not None and text_ref_vocab_size is not None and policy_vocab_size != text_ref_vocab_size:
            raise ValueError(
                f'text_ref_model vocab_size ({text_ref_vocab_size}) must match current model vocab_size '
                f'({policy_vocab_size})')

    def _init_grpo_params(self):
        super()._init_grpo_params()
        args = self.args
        self.eta = args.eta
        self.diff_objective = args.diff_objective
        self.diff_kl_type = args.diff_kl_type
        self.diff_margin_gamma = args.diff_margin_gamma
        self.diff_margin_beta = args.diff_margin_beta
        self.entropy_mask_quantile = args.entropy_mask_quantile
        self.entropy_mask_scope = args.entropy_mask_scope
        self.entropy_mask_type = args.entropy_mask_type
        self.text_only_modality_scope = args.text_only_modality_scope

        # MAPO uses text-reference entropy mask. Disable GRPO entropy mask path.
        self.top_entropy_quantile = 1.0
        self.compute_entropy = self.log_entropy
        self.text_ref_models = []

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        args = get_args()
        if args.text_ref_model or args.text_ref_load:
            text_ref_model = args.text_ref_model or args.model
            if args.text_ref_model:
                text_ref_info, _ = get_model_info_meta(
                    args.text_ref_model,
                    model_type=args.model_type,
                    use_hf=args.use_hf,
                    hub_token=args.hub_token)
                self._validate_text_ref_compatibility(args, text_ref_info)
                text_ref_model = text_ref_info.model_dir

            text_ref_models = get_model(model_provider_func, model_type, wrap_with_ddp=False)
            if args.text_ref_load is None:
                for model in text_ref_models:
                    model = unwrap_model(model)
                    self.bridge.load_weights(model, text_ref_model)
                    model.requires_grad_(False).eval()
            else:
                for model in text_ref_models:
                    model = unwrap_model(model)
                    model.requires_grad_(False).eval()
                load_checkpoint(text_ref_models, None, None, load_arg='text_ref_load')
            self.text_ref_models = text_ref_models
        return super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

    @contextmanager
    def _text_ref_context(self):
        if self.text_ref_models:
            with ExitStack() as stack:
                for model in self.text_ref_models:
                    unwrapped_model = unwrap_model(model)
                    if hasattr(unwrapped_model, 'disable_adapter'):
                        stack.enter_context(unwrapped_model.disable_adapter())
                yield self.text_ref_models
        else:
            with self.null_ref_context() as ref_models:
                yield ref_models

    def _prepare_model_inputs(self, inputs):
        model_inputs = super()._prepare_model_inputs(inputs)
        for key in ['text_ref_per_token_logps', 'text_ref_per_token_entropy', 'prompt_group_ids']:
            model_inputs.pop(key, None)
        return model_inputs

    def _get_encoded_batch(self, encoded_list, rollout_batch, template):
        encoded_batch = super()._get_encoded_batch(encoded_list, rollout_batch, template)

        prompt_ids = [item.get('prompt_id', f'prompt_{i}') for i, item in enumerate(rollout_batch)]
        prompt_id_to_group = {}
        prompt_group_ids = []
        for prompt_id in prompt_ids:
            if prompt_id not in prompt_id_to_group:
                prompt_id_to_group[prompt_id] = len(prompt_id_to_group)
            prompt_group_ids.append(prompt_id_to_group[prompt_id])
        encoded_batch['prompt_group_ids'] = torch.tensor(prompt_group_ids, dtype=torch.long, device=self.device)
        return encoded_batch

    def _build_text_only_inputs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        text_inputs = deepcopy(model_inputs)
        audio_keys = {
            'audios', 'input_features', 'feature_attention_mask', 'input_features_mask', 'speech', 'speech_lengths',
            'audio_values', 'audio_attention_mask'
        }
        multimodal_keys = audio_keys | {
            'images', 'videos', 'objects', 'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw',
            'image_grid_hws', 'video_second_per_grid', 'cross_images', 'multimodal'
        }
        drop_keys = audio_keys if self.text_only_modality_scope == 'audio' else multimodal_keys
        for key in drop_keys:
            text_inputs.pop(key, None)
        return text_inputs

    def _compute_per_token_logps_and_entropy(self, model, model_inputs: Dict[str, Any], batch_size: int,
                                             max_seq_len: int,
                                             seq_lengths: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        args = get_args()
        labels = model_inputs.get('labels')
        if labels is None:
            return None, None

        packed_seq_params = model_inputs.get('packed_seq_params')
        inputs_for_logits = {k: v for k, v in model_inputs.items() if k != 'labels'}
        output_tensor = forward_step_helper(model, inputs_for_logits)
        if output_tensor is None:
            return None, None

        per_token_logps_raw, per_token_entropy_raw = compute_logps_and_entropy_from_logits(
            output_tensor, labels, compute_entropy=True)

        if args.context_parallel_size > 1:
            num_samples = packed_seq_params.num_samples if args.padding_free and packed_seq_params is not None else batch_size
            per_token_logps_raw = self._postprocess_packed_tensor_cp(per_token_logps_raw, packed_seq_params, num_samples)
            per_token_entropy_raw = self._postprocess_packed_tensor_cp(per_token_entropy_raw, packed_seq_params, num_samples)

        if args.padding_free:
            per_token_logps, _ = pad_logps_back_to_batch(
                logps_rmpad=per_token_logps_raw,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths)
            per_token_entropy, _ = pad_logps_back_to_batch(
                logps_rmpad=per_token_entropy_raw,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths,
                pad_value=float('nan'))
        else:
            per_token_logps = per_token_logps_raw
            per_token_entropy = per_token_entropy_raw

        return per_token_logps, per_token_entropy

    def _maybe_compute_logps(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = super()._maybe_compute_logps(batch)

        model_inputs = self._prepare_model_inputs(batch)
        text_only_inputs = self._build_text_only_inputs(model_inputs)
        seq_lengths = batch['seq_lengths']
        batch_size = batch['num_samples']
        max_seq_len = batch['completion_mask'].shape[1]

        with torch.no_grad(), self._text_ref_context() as text_ref_models:
            assert len(text_ref_models) == 1, 'MAPO currently does not support VPP.'
            text_ref_model = text_ref_models[0]
            text_ref_per_token_logps, text_ref_per_token_entropy = self._compute_per_token_logps_and_entropy(
                text_ref_model, text_only_inputs, batch_size=batch_size, max_seq_len=max_seq_len, seq_lengths=seq_lengths)
            batch['text_ref_per_token_logps'] = text_ref_per_token_logps
            batch['text_ref_per_token_entropy'] = text_ref_per_token_entropy
        return batch

    def _compute_diff_kl(self, per_token_logps: torch.Tensor, text_ref_per_token_logps: torch.Tensor) -> torch.Tensor:
        forward_kl = torch.exp(text_ref_per_token_logps - per_token_logps) - (text_ref_per_token_logps - per_token_logps) - 1
        if self.diff_kl_type == 'forward':
            return forward_kl
        reverse_kl = torch.exp(per_token_logps - text_ref_per_token_logps) - (per_token_logps - text_ref_per_token_logps) - 1
        if self.diff_kl_type == 'reverse':
            return reverse_kl
        return 0.5 * (forward_kl + reverse_kl)

    def _compute_diff_objective(self, per_token_logps: torch.Tensor,
                                text_ref_per_token_logps: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta = per_token_logps - text_ref_per_token_logps
        if self.diff_objective == 'softplus_margin':
            if self.diff_kl_type == 'forward':
                # Forward-style direction: penalize delta > -gamma.
                margin_gap = self.diff_margin_gamma + delta
                margin_loss = torch.nn.functional.softplus(self.diff_margin_beta * margin_gap) / self.diff_margin_beta
            elif self.diff_kl_type == 'reverse':
                # Reverse-style direction: penalize delta < gamma.
                margin_gap = self.diff_margin_gamma - delta
                margin_loss = torch.nn.functional.softplus(self.diff_margin_beta * margin_gap) / self.diff_margin_beta
            else:
                # Symmetric direction: average forward/reverse margin penalties.
                reverse_gap = self.diff_margin_gamma - delta
                forward_gap = self.diff_margin_gamma + delta
                reverse_loss = torch.nn.functional.softplus(self.diff_margin_beta * reverse_gap) / self.diff_margin_beta
                forward_loss = torch.nn.functional.softplus(self.diff_margin_beta * forward_gap) / self.diff_margin_beta
                margin_loss = 0.5 * (reverse_loss + forward_loss)
            return {'adjustment': margin_loss, 'margin_loss': margin_loss, 'delta': delta}

        diff_kl = self._compute_diff_kl(per_token_logps, text_ref_per_token_logps)
        # Keep backward-compatible behavior for KL objectives:
        # objective uses per_token_pg_loss - eta * diff_kl.
        return {'adjustment': -diff_kl, 'diff_kl': diff_kl, 'delta': delta}

    @staticmethod
    def _safe_quantile(values: torch.Tensor, quantile: float) -> Optional[torch.Tensor]:
        valid = values[~torch.isnan(values)]
        if valid.numel() == 0:
            return None
        return torch.quantile(valid.float(), quantile)

    def _build_mapo_entropy_weights(
            self,
            entropies: torch.Tensor,
            completion_mask: torch.Tensor,
            prompt_group_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        completion_mask = completion_mask.bool()
        completion_mask_f = completion_mask.float()

        if self.entropy_mask_type == 'hard':
            if self.entropy_mask_quantile >= 1.0:
                keep_mask = completion_mask.clone()
                kept_frac = torch.tensor(1.0, device=completion_mask.device)
                return keep_mask.float(), {
                    'entropy_threshold': torch.tensor(float('nan'), device=completion_mask.device),
                    'entropy_kept_frac': kept_frac
                }

            keep_mask = torch.zeros_like(completion_mask, dtype=torch.bool)
            thresholds = []
            quantile = 1 - self.entropy_mask_quantile

            if self.entropy_mask_scope == 'sequence':
                for i in range(entropies.shape[0]):
                    threshold = self._safe_quantile(entropies[i], quantile)
                    if threshold is None:
                        continue
                    thresholds.append(threshold)
                    keep_mask[i] = entropies[i] >= threshold
            elif self.entropy_mask_scope == 'group' and prompt_group_ids is not None:
                for group_id in prompt_group_ids.unique():
                    row_mask = prompt_group_ids == group_id
                    threshold = self._safe_quantile(entropies[row_mask].flatten(), quantile)
                    if threshold is None:
                        continue
                    thresholds.append(threshold)
                    keep_mask[row_mask] = entropies[row_mask] >= threshold
            else:
                threshold = self._safe_quantile(entropies.flatten(), quantile)
                if threshold is not None:
                    thresholds.append(threshold)
                    keep_mask = entropies >= threshold

            keep_mask = keep_mask & completion_mask
            threshold_mean = torch.stack(thresholds).mean() if thresholds else torch.tensor(
                float('nan'), device=entropies.device)
            kept_frac = keep_mask.float().sum() / completion_mask_f.sum().clamp(min=1.0)
            return keep_mask.float(), {'entropy_threshold': threshold_mean, 'entropy_kept_frac': kept_frac}

        # soft mode: normalize entropy weights within the configured scope.
        # All completion tokens are retained; low-entropy tokens receive smaller weights.
        scores = torch.nan_to_num(entropies.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.clamp(scores, min=0.0) + completion_mask_f * 1e-12
        weights = torch.zeros_like(scores)

        def _normalize_partition(partition_mask: torch.Tensor):
            token_mask = partition_mask & completion_mask
            token_count = token_mask.sum()
            if token_count.item() == 0:
                return
            token_scores = scores * token_mask.float()
            denom = token_scores.sum()
            denom_is_valid = torch.isfinite(denom).item() and denom.item() > 0
            if denom_is_valid:
                weights[token_mask] = token_scores[token_mask] / denom
            else:
                weights[token_mask] = 1.0 / token_count.float()

        if self.entropy_mask_scope == 'sequence':
            for i in range(scores.shape[0]):
                partition_mask = torch.zeros_like(completion_mask, dtype=torch.bool)
                partition_mask[i] = True
                _normalize_partition(partition_mask)
        elif self.entropy_mask_scope == 'group' and prompt_group_ids is not None:
            for group_id in prompt_group_ids.unique():
                row_mask = prompt_group_ids == group_id
                partition_mask = row_mask.unsqueeze(-1).expand_as(completion_mask)
                _normalize_partition(partition_mask)
        else:
            _normalize_partition(torch.ones_like(completion_mask, dtype=torch.bool))

        kept_frac = torch.tensor(1.0, device=entropies.device)
        return weights, {
            'entropy_threshold': torch.tensor(float('nan'), device=entropies.device),
            'entropy_kept_frac': kept_frac
        }

    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
        args = get_args()
        advantages = data['advantages']
        completion_mask = data['completion_mask']
        packed_seq_params = data.get('packed_seq_params')
        truncated_mask = data['truncated_mask']
        seq_lengths = data['seq_lengths']
        micro_batch_size = self.micro_batch_size

        per_token_logps = data.get('per_token_logps')
        ref_per_token_logps = data.get('ref_per_token_logps')
        old_per_token_logps = data.get('old_per_token_logps')
        rollout_per_token_logps = data.get('rollout_per_token_logps')
        text_ref_per_token_logps = data.get('text_ref_per_token_logps')
        text_ref_per_token_entropy = data.get('text_ref_per_token_entropy')
        prompt_group_ids = data.get('prompt_group_ids')

        if args.padding_free:
            lengths = packed_seq_params.cu_seqlens_q[1:micro_batch_size
                                                     + 1] - packed_seq_params.cu_seqlens_q[:micro_batch_size]
        else:
            lengths = seq_lengths

        rollout_correction_metrics = {}
        should_compute_rollout_metrics = (
            self.rollout_importance_sampling_mode is not None or self.log_rollout_offpolicy_metrics)
        local_has_rollout_per_token_logps = rollout_per_token_logps is not None
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_has_rollout_per_token_logps = gather_object([local_has_rollout_per_token_logps], group=dp_group)
        should_compute_rollout_metrics = should_compute_rollout_metrics and all(all_has_rollout_per_token_logps)
        if (not self.disable_rollout_importance_sampling and should_compute_rollout_metrics):
            rollout_correction_metrics = self._compute_rollout_offpolicy_metrics(old_per_token_logps,
                                                                                 rollout_per_token_logps,
                                                                                 completion_mask)
            if self.rollout_importance_sampling_mode is not None:
                rollout_log_ratio = old_per_token_logps - rollout_per_token_logps
                rollout_is_weights = self._apply_rollout_importance_sampling(rollout_log_ratio, completion_mask)
                is_metrics = self._compute_is_correction_metrics(rollout_log_ratio, rollout_is_weights, completion_mask)
                rollout_correction_metrics.update(is_metrics)

        if self.args.overlong_filter and truncated_mask.any():
            if truncated_mask.all():
                logger.warning('All completions are truncated in this batch. Loss and grad_norm will be 0. '
                               'Consider increasing max_completion_length')
            truncated_mask_expanded = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask_expanded)

        if self.beta != 0.0 and ref_per_token_logps is not None and not self.kl_in_reward:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        else:
            per_token_kl = None

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            seq_level_log_weights = ((log_ratio * completion_mask).sum(-1)
                                     / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                ",'sequence' and 'sequence_token'.")
        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_pg_loss = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
        elif self.loss_type == 'sapo':
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1)) * (4.0 / self.tau_pos)
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1)) * (4.0 / self.tau_neg)
            is_positive = advantages.unsqueeze(1) > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)
            per_token_pg_loss = -soft_gate * advantages.unsqueeze(1)
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_pg_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        if self.rollout_importance_sampling_mode is not None:
            per_token_pg_loss = per_token_pg_loss * rollout_is_weights

        diff_objective_terms = None
        if self.eta != 0.0 and text_ref_per_token_logps is not None:
            diff_objective_terms = self._compute_diff_objective(per_token_logps, text_ref_per_token_logps)

        # PG and diff-KL are normalized with this weight tensor.
        pg_weights = completion_mask.float()
        mapo_entropy_metrics = {}
        if text_ref_per_token_entropy is not None:
            text_entropies = text_ref_per_token_entropy.masked_fill(completion_mask == 0, float('nan'))
            pg_weights, entropy_metrics = self._build_mapo_entropy_weights(
                text_entropies, completion_mask, prompt_group_ids=prompt_group_ids)
            mapo_entropy_metrics = {
                'mapo/entropy_threshold': entropy_metrics['entropy_threshold'],
                'mapo/entropy_kept_frac': entropy_metrics['entropy_kept_frac']
            }

        per_token_pg_obj = per_token_pg_loss
        if diff_objective_terms is not None:
            per_token_pg_obj = per_token_pg_obj + self.eta * diff_objective_terms['adjustment']

        if self.off_policy_sequence_mask_delta is not None:
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, advantages)
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded
            pg_weights = pg_weights * off_policy_seq_mask_expanded.float()

        if self.loss_type in ['grpo', 'sapo']:
            pg_loss = ((per_token_pg_obj * pg_weights).sum(-1) / pg_weights.sum(-1).clamp(min=1.0)).mean()
            if self.beta != 0.0 and per_token_kl is not None:
                reg_loss = ((per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
                loss = pg_loss + self.beta * reg_loss
            else:
                loss = pg_loss
        elif self.loss_type in ['bnpo', 'dr_grpo', 'cispo', 'dapo']:
            pg_loss = (per_token_pg_obj * pg_weights).sum() / pg_weights.sum().clamp(min=1.0)
            if self.beta != 0.0 and per_token_kl is not None:
                reg_loss = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                loss = pg_loss + self.beta * reg_loss
            else:
                loss = pg_loss
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        avg_metric = {'loss': loss.clone().detach()}
        custom_metrics = {}
        total_lengths = gather(lengths, group=mpu.get_data_parallel_group(with_context_parallel=True))
        custom_metrics = {
            'completions/mean_length': total_lengths.float().mean(),
            'completions/max_length': total_lengths.float().max(),
            'completions/min_length': total_lengths.float().min(),
            'mapo/eta': torch.tensor(self.eta, device=loss.device),
            'mapo/diff_objective_is_softplus': torch.tensor(
                1.0 if self.diff_objective == 'softplus_margin' else 0.0, device=loss.device),
        }
        if self.diff_objective == 'softplus_margin':
            custom_metrics['mapo/diff_margin_gamma'] = torch.tensor(self.diff_margin_gamma, device=loss.device)
            custom_metrics['mapo/diff_margin_beta'] = torch.tensor(self.diff_margin_beta, device=loss.device)

        for key, value in mapo_entropy_metrics.items():
            custom_metrics[key] = value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(
                value, device=loss.device)

        if self.beta != 0.0 and per_token_kl is not None:
            kl_value = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            avg_metric['kl'] = kl_value.clone().detach()

        if diff_objective_terms is not None:
            weighted_denom = pg_weights.sum().clamp(min=1.0)
            if 'diff_kl' in diff_objective_terms:
                diff_kl_value = (diff_objective_terms['diff_kl'] * pg_weights).sum() / weighted_denom
                custom_metrics['mapo/diff_kl'] = diff_kl_value.clone().detach()
            else:
                margin_loss_value = (diff_objective_terms['margin_loss'] * pg_weights).sum() / weighted_denom
                custom_metrics['mapo/diff_margin_loss'] = margin_loss_value.clone().detach()
                # Keep metric key compatibility for dashboards that already track mapo/diff_kl.
                custom_metrics['mapo/diff_kl'] = margin_loss_value.clone().detach()
            delta_value = (diff_objective_terms['delta'] * pg_weights).sum() / weighted_denom
            custom_metrics['mapo/diff_delta'] = delta_value.clone().detach()

        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        if self.loss_type == 'cispo':
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = (is_cispo_clipped.float() * completion_mask).sum() / completion_token_count
            self._metrics[mode]['cispo_clip_ratio'].append(cispo_clip_ratio)
        elif self.loss_type == 'sapo':
            pass
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_1_for_metrics = torch.exp(log_importance_weights)
            is_low_clipped = (coef_1_for_metrics < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1_for_metrics > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            low_clip = (is_low_clipped.float() * completion_mask).sum() / completion_token_count
            high_clip = (is_high_clipped.float() * completion_mask).sum() / completion_token_count
            is_region_clipped = is_low_clipped | is_high_clipped
            clip_ratio = (is_region_clipped.float() * completion_mask).sum() / completion_token_count

            gathered_low_clip = gather(
                low_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            gathered_high_clip = gather(
                high_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            self._metrics[mode]['clip_ratio/low_mean'].append(low_clip.item())
            self._metrics[mode]['clip_ratio/high_mean'].append(high_clip.item())
            self._metrics[mode]['clip_ratio/region_mean'].append(clip_ratio.item())
            custom_metrics['clip_ratio/low_min'] = gathered_low_clip.min()
            custom_metrics['clip_ratio/high_max'] = gathered_high_clip.max()

        if rollout_correction_metrics:
            for key, value in rollout_correction_metrics.items():
                if isinstance(value, torch.Tensor):
                    custom_metrics[f'rollout_correction/{key}'] = value.clone().detach()
                else:
                    custom_metrics[f'rollout_correction/{key}'] = torch.tensor(value, device=loss.device)

        if self._metrics[mode]:
            addition_metrics = {
                key: torch.tensor(sum(val) / len(val), device=loss.device)
                for key, val in self._metrics[mode].items()
            }
            avg_metric.update(addition_metrics)

        avg_metric = self._all_reduce_metric(avg_metric)
        reporting_metric = {**avg_metric, **custom_metrics}

        if (self.log_completions and self.is_main_process and (self._step - 1) % self.steps_per_generation == 0
                and self._step != self._last_logged_step):
            table = {
                'gen_step': [self._step - 1] * len(self._logs['prompt']),
                'prompt': list(self._logs['prompt']),
                'completion': list(self._logs['completion']),
                'advantages': list(self._logs['advantages']),
            }
            for reward_func_name in self._logs['rewards'].keys():
                table[reward_func_name] = list(self._logs['rewards'][reward_func_name])
            import pandas as pd
            df = pd.DataFrame(table)
            if self.wandb_log_unique_prompts:
                df = df.drop_duplicates(subset=['prompt'])
            wandb_writer = get_wandb_writer()
            if wandb_writer is not None:
                import wandb
                wandb_writer.log({'completions': wandb.Table(dataframe=df)})
            elif args.report_to == 'swanlab':
                import swanlab
                headers = list(table.keys())
                rows = []
                for i in range(len(table['gen_step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
            self._last_logged_step = self._step

        return loss, reporting_metric
