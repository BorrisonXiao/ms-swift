# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

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
from .mapo_attention_collector import (MAPOAttentionCollector, build_audio_token_mask, detect_audio_token_ids,
                                       parse_attention_layer_spec)
from .mapo_pos_utils import build_pos_token_gate, normalize_target_pos_tags
from .utils import gather, gather_object
from .vocab_parallel_utils import compute_logps_and_entropy_from_logits

logger = get_logger()
_LAYER_INDEX_PATTERNS = (
    re.compile(r'\.layers\.(\d+)(?:\.|$)'),
    re.compile(r'\.layer\.(\d+)(?:\.|$)'),
    re.compile(r'\.h\.(\d+)(?:\.|$)'),
)
_MAPO_ATTN_LOG_EPS = 1e-6


class MegatronMAPOTrainer(MegatronGRPOTrainer):
    _DEPRECATED_ARG_DEFAULTS = {
        'diff_objective': 'softplus_margin',
        'diff_kl_type': 'forward',
        'diff_margin_gamma': 2.0,
        'diff_margin_beta': 4.0,
        'entropy_mask_quantile': 0.25,
        'entropy_mask_scope': 'sequence',
        'entropy_mask_type': 'hard',
    }

    @staticmethod
    def _validate_text_ref_compatibility(args, text_ref_info):
        """Validate text-reference model compatibility with the policy model config."""
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

    def _warn_deprecated_mapo_args(self):
        """Warn once when deprecated phase-1 MAPO arguments are set to non-default values."""
        ignored_args = []
        for key, default_value in self._DEPRECATED_ARG_DEFAULTS.items():
            value = getattr(self.args, key, default_value)
            if value != default_value:
                ignored_args.append(f'{key}={value}')
        if ignored_args:
            logger.warning(
                'MAPO phase-2 ignores deprecated MAPO phase-1 args: %s. '
                'These arguments are parser-compatible for one deprecation cycle.',
                ', '.join(ignored_args))

    def _should_log_vs_samples(self, key: str) -> bool:
        return key.startswith('rewards/')

    def training_log(self, loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad):
        if getattr(self, 'mapo_debug_attn_grad_probe', False):
            device = None
            for value in loss_dict.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            loss_dict.update(self._finalize_mapo_attn_probe_for_logging(device))
        return super().training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                                    loss_scale, report_memory_flag, skipped_iter, grad_norm, params_norm,
                                    num_zeros_in_grad)

    @staticmethod
    def _parse_mapo_pos_tags(tag_spec: Any) -> Set[str]:
        """Normalize user POS-tag input into canonical uppercase POS tags used by MAPO."""
        if isinstance(tag_spec, str):
            values = [value.strip() for value in tag_spec.split(',')]
        elif isinstance(tag_spec, (list, tuple, set)):
            values = [str(value).strip() for value in tag_spec]
        else:
            values = [str(tag_spec).strip()]
        return normalize_target_pos_tags(values)

    @staticmethod
    def _flatten_response_token_ids(token_ids: Any) -> List[int]:
        """Flatten nested response token-id containers into a plain list of ints."""
        if token_ids is None:
            return []
        if isinstance(token_ids, torch.Tensor):
            return [int(v) for v in token_ids.detach().view(-1).tolist()]
        if isinstance(token_ids, (list, tuple)):
            flattened: List[int] = []
            for item in token_ids:
                flattened.extend(MegatronMAPOTrainer._flatten_response_token_ids(item))
            return flattened
        try:
            return [int(token_ids)]
        except Exception:
            return []

    @staticmethod
    def _align_pos_gate_values(gate_values: Any,
                               token_count: int,
                               device: torch.device) -> Tuple[torch.Tensor, bool]:
        """Align per-sample POS gate values to token_count with robust fallbacks."""
        if token_count <= 0:
            return torch.zeros((0, ), dtype=torch.float32, device=device), False

        if isinstance(gate_values, torch.Tensor):
            values = [float(v) for v in gate_values.detach().view(-1).tolist()]
        elif isinstance(gate_values, (list, tuple)):
            values = [float(v) for v in gate_values]
        else:
            values = []

        fallback_used = False
        if len(values) < token_count:
            fallback_used = True
            values = [1.0] * token_count
        elif len(values) > token_count:
            values = values[:token_count]

        gate = torch.tensor(values, dtype=torch.float32, device=device)
        if not torch.isfinite(gate).all():
            fallback_used = True
            gate = torch.ones((token_count, ), dtype=torch.float32, device=device)
        gate = gate.clamp(min=0.0, max=1.0)
        if gate.sum().item() <= 0.0:
            fallback_used = True
            gate = torch.ones((token_count, ), dtype=torch.float32, device=device)
        return gate, fallback_used

    @staticmethod
    def _parse_layer_index(module_name: str) -> Optional[int]:
        """Parse transformer layer index from a module name string."""
        name = str(module_name)
        for pattern in _LAYER_INDEX_PATTERNS:
            match = pattern.search(name)
            if match is None:
                continue
            try:
                return int(match.group(1))
            except Exception:
                continue
        return None

    @staticmethod
    def _is_qwen_thinker_text_attention(module_name: str, module) -> bool:
        """Return whether a module corresponds to Qwen thinker text self-attention."""
        name = str(module_name).lower()
        if (
            name.endswith('.self_attention')
            or name.endswith('.self_attn')
            or name == 'self_attention'
            or name == 'self_attn'
        ):
            return True
        if module.__class__.__name__ == 'Qwen3OmniMoeThinkerTextAttention':
            return True
        return False

    @staticmethod
    def _compute_mapo_attention_probs_from_qk(attn_module, query: torch.Tensor, key: torch.Tensor,
                                              attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Reconstruct attention probabilities from query/key tensors for MAPO capture."""
        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            return None
        # TE/Megatron may emit either:
        # - 4D: [s, b, h, d]
        # - 3D (packed): [t, h, d]
        if query.ndim == 3 and key.ndim == 3:
            query = query.unsqueeze(1)  # [t, 1, h, d]
            key = key.unsqueeze(1)      # [t, 1, g, d]
        elif query.ndim != 4 or key.ndim != 4:
            return None

        try:
            key_layer = key
            num_heads = int(getattr(attn_module, 'num_attention_heads_per_partition', 0) or 0)
            num_groups = int(getattr(attn_module, 'num_query_groups_per_partition', 0) or 0)
            if num_heads > 0 and num_groups > 0 and (num_heads // num_groups) > 1:
                key_layer = key.repeat_interleave(num_heads // num_groups, dim=2)

            output_size = (query.size(1), query.size(2), query.size(0), key_layer.size(0))
            query_2d = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
            key_2d = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

            scale = None
            core_attention = getattr(attn_module, 'core_attention', None)
            if core_attention is not None:
                scale = getattr(core_attention, 'softmax_scale', None)
            if scale is None:
                head_dim = max(int(query_2d.size(-1)), 1)
                scale = head_dim**-0.5

            attn_scores = torch.bmm(
                query_2d.transpose(0, 1), key_2d.transpose(0, 1).transpose(1, 2)) * float(scale)
            attn_scores = attn_scores.view(*output_size)

            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
                attn_scores = attn_scores + attention_mask[:, :, :output_size[2], :output_size[3]]

            return torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        except Exception:
            return None

    def _patch_mapo_core_attention_forward(self, attn_module) -> bool:
        """Patch one attention module to cache attention probabilities during forward."""
        core_attention = getattr(attn_module, 'core_attention', None)
        if core_attention is None or not hasattr(core_attention, 'forward'):
            return False
        if bool(getattr(core_attention, '_mapo_core_forward_patched', False)):
            return True

        original_forward = core_attention.forward

        def _wrapped_forward(*args, **kwargs):
            """Capture Q/K/mask diagnostics and cached attention probs, then call original forward."""
            try:
                setattr(
                    attn_module, '_mapo_core_forward_calls',
                    int(getattr(attn_module, '_mapo_core_forward_calls', 0)) + 1)
                query = args[0] if len(args) > 0 else kwargs.get('query')
                key = args[1] if len(args) > 1 else kwargs.get('key')
                attention_mask = args[3] if len(args) > 3 else kwargs.get('attention_mask')
                setattr(attn_module, '_mapo_core_last_query_shape',
                        tuple(query.shape) if isinstance(query, torch.Tensor) else None)
                setattr(attn_module, '_mapo_core_last_key_shape',
                        tuple(key.shape) if isinstance(key, torch.Tensor) else None)
                setattr(attn_module, '_mapo_core_last_mask_shape',
                        tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None)
                attn_module._mapo_last_attention_probs = self._compute_mapo_attention_probs_from_qk(
                    attn_module=attn_module, query=query, key=key, attention_mask=attention_mask)
                if isinstance(attn_module._mapo_last_attention_probs, torch.Tensor):
                    setattr(
                        attn_module, '_mapo_core_forward_success',
                        int(getattr(attn_module, '_mapo_core_forward_success', 0)) + 1)
                    setattr(attn_module, '_mapo_last_attention_error', '')
                    if not bool(getattr(attn_module, '_mapo_core_debug_logged_once', False)):
                        logger.info(
                            'MAPO core attention capture first success: module=%s q_shape=%s k_shape=%s attn_shape=%s',
                            getattr(attn_module, '_mapo_name', attn_module.__class__.__name__),
                            getattr(attn_module, '_mapo_core_last_query_shape', None),
                            getattr(attn_module, '_mapo_core_last_key_shape', None),
                            tuple(attn_module._mapo_last_attention_probs.shape))
                        setattr(attn_module, '_mapo_core_debug_logged_once', True)
                else:
                    error_msg = (
                        f'capture_returned_none(query_shape={getattr(attn_module, "_mapo_core_last_query_shape", None)}, '
                        f'key_shape={getattr(attn_module, "_mapo_core_last_key_shape", None)}, '
                        f'mask_shape={getattr(attn_module, "_mapo_core_last_mask_shape", None)})')
                    setattr(attn_module, '_mapo_last_attention_error', error_msg)
                    if not bool(getattr(attn_module, '_mapo_core_error_logged_once', False)):
                        logger.warning(
                            'MAPO core attention capture returned None: module=%s %s',
                            getattr(attn_module, '_mapo_name', attn_module.__class__.__name__), error_msg)
                        setattr(attn_module, '_mapo_core_error_logged_once', True)
            except Exception:
                attn_module._mapo_last_attention_probs = None
                setattr(attn_module, '_mapo_last_attention_error', 'capture_exception')
            return original_forward(*args, **kwargs)

        core_attention.forward = _wrapped_forward
        core_attention._mapo_core_forward_patched = True
        core_attention._mapo_original_forward = original_forward
        return True

    def _configure_mapo_local_eager_override(self, model, target_layers: Set[int]) -> int:
        """Enable MAPO eager capture on selected local attention layers."""
        override_count = 0
        override_debug: List[Dict[str, Any]] = []
        override_modules = []
        for module_name, module in model.named_modules():
            if not self._is_qwen_thinker_text_attention(module_name, module):
                continue
            layer_idx = self._parse_layer_index(module_name)
            if layer_idx is None:
                continue
            use_eager = (layer_idx in target_layers)
            original_checkpoint_core_attention = getattr(module, '_mapo_original_checkpoint_core_attention', None)
            if original_checkpoint_core_attention is None and hasattr(module, 'checkpoint_core_attention'):
                original_checkpoint_core_attention = bool(getattr(module, 'checkpoint_core_attention'))
                setattr(module, '_mapo_original_checkpoint_core_attention', original_checkpoint_core_attention)
            setattr(module, '_mapo_force_eager', bool(use_eager))
            if hasattr(module, 'checkpoint_core_attention'):
                if use_eager:
                    setattr(module, 'checkpoint_core_attention', False)
                elif original_checkpoint_core_attention is not None:
                    setattr(module, 'checkpoint_core_attention', bool(original_checkpoint_core_attention))
            setattr(module, '_mapo_last_attention_probs', None)
            setattr(module, '_mapo_last_attention_error', '')
            setattr(module, '_mapo_core_forward_calls', 0)
            setattr(module, '_mapo_core_forward_success', 0)
            setattr(module, '_mapo_core_last_query_shape', None)
            setattr(module, '_mapo_core_last_key_shape', None)
            setattr(module, '_mapo_core_last_mask_shape', None)
            setattr(module, '_mapo_name', str(module_name))
            if use_eager:
                patched_core_forward = self._patch_mapo_core_attention_forward(module)
                override_count += 1
                override_modules.append(module)
                if len(override_debug) < 16:
                    override_debug.append({
                        'module_name': str(module_name),
                        'module_class': module.__class__.__name__,
                        'layer_idx': int(layer_idx),
                        'has_flag': bool(hasattr(module, '_mapo_force_eager')),
                        'flag_value': bool(getattr(module, '_mapo_force_eager', False)),
                        'checkpoint_core_attention': bool(getattr(module, 'checkpoint_core_attention', False)),
                        'core_attention_class': getattr(getattr(module, 'core_attention', None), '__class__',
                                                        type(None)).__name__,
                        'core_forward_patched': bool(patched_core_forward),
                    })
        self._mapo_eager_override_debug = override_debug
        self._mapo_target_attention_modules = override_modules
        return override_count

    @staticmethod
    def _masked_temperature_softmax(scores: torch.Tensor,
                                    completion_mask: torch.Tensor,
                                    temperature: Any,
                                    gate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute masked temperature-softmax over valid completion tokens.

        Supports scalar or per-sequence temperatures. `gate_mask` can further
        restrict the active tokens (for example POS gating on nu-branch).
        """
        completion_mask = completion_mask.bool()
        completion_mask_f = completion_mask.float()
        if isinstance(temperature, torch.Tensor):
            temp = torch.nan_to_num(temperature.float(), nan=1.0, posinf=1.0, neginf=1.0).to(device=scores.device)
            if temp.ndim == 1:
                temp = temp.unsqueeze(-1)
            temp = temp.clamp(min=1e-6)
        else:
            temp = max(float(temperature), 1e-6)
        if gate_mask is None:
            active_mask_f = completion_mask_f
        else:
            active_mask_f = torch.nan_to_num(gate_mask.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if active_mask_f.shape != completion_mask_f.shape:
                active_mask_f = completion_mask_f
            else:
                active_mask_f = active_mask_f.clamp(min=0.0, max=1.0) * completion_mask_f
        active_mask = active_mask_f > 0
        masked_scores = scores.float().masked_fill(~active_mask, float('-inf')) / temp
        row_max = masked_scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_scores = torch.exp(masked_scores - row_max) * active_mask_f
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return exp_scores / denom

    @staticmethod
    def _build_length_scaled_temperature(completion_mask: torch.Tensor, base_temperature: float) -> torch.Tensor:
        """Build per-sequence temperature tau_T = tau_base * log(T_i), with safe clamping."""
        completion_mask_f = completion_mask.float()
        token_count = completion_mask_f.sum(-1, keepdim=True).clamp(min=1.0)
        # Use log(max(T_i, 2)) to avoid degenerate tau=0 when T_i is 1.
        tau_t = float(base_temperature) * torch.log(token_count.clamp(min=2.0))
        return tau_t.clamp(min=1e-6)

    def _build_aligned_audio_token_mask(self, encoded_batch: Dict[str, Any],
                                        completion_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Build and align audio-token mask to the encoded completion tensor shape."""
        input_ids = encoded_batch.get('input_ids')
        audio_token_mask = build_audio_token_mask(input_ids, self._mapo_audio_token_ids)
        if not isinstance(audio_token_mask, torch.Tensor) or audio_token_mask.ndim != 2:
            return None

        batch_size, seq_len = int(completion_mask.shape[0]), int(completion_mask.shape[1])
        audio_token_mask = audio_token_mask.to(device=completion_mask.device, dtype=torch.bool)
        if audio_token_mask.shape == completion_mask.shape:
            return audio_token_mask

        # Padding-free mode: [1, total_tokens] -> [batch, max_seq_len].
        if audio_token_mask.shape[0] == 1:
            seq_lengths = encoded_batch.get('seq_lengths')
            if isinstance(seq_lengths, torch.Tensor) and seq_lengths.ndim == 1 and seq_lengths.shape[0] == batch_size:
                try:
                    audio_mask_padded, _ = pad_logps_back_to_batch(
                        logps_rmpad=audio_token_mask.float(),
                        logits_to_keep=seq_len,
                        batch_size=batch_size,
                        seq_lengths=seq_lengths,
                        pad_value=0.0)
                    return audio_mask_padded > 0.5
                except Exception:
                    pass

        audio_mask_padded = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=completion_mask.device)
        b = min(batch_size, int(audio_token_mask.shape[0]))
        q = min(seq_len, int(audio_token_mask.shape[1]))
        audio_mask_padded[:b, :q] = audio_token_mask[:b, :q]
        return audio_mask_padded

    @staticmethod
    def _reduce_token_objective(per_token_obj: torch.Tensor,
                                completion_mask: torch.Tensor,
                                loss_type: str,
                                denom_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reduce a per-token MAPO objective using the same aggregation style as GRPO loss_type.

        - grpo/sapo: sequence-normalized first, then average over batch.
        - bnpo/dr_grpo/cispo/dapo: token-normalized over the whole micro-batch.
        """
        completion_mask_f = completion_mask.float()
        if denom_weights is None:
            denom_weights_f = completion_mask_f
        else:
            denom_weights_f = torch.nan_to_num(denom_weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if denom_weights_f.shape != completion_mask_f.shape:
                denom_weights_f = completion_mask_f
            else:
                denom_weights_f = denom_weights_f * completion_mask_f
        if loss_type in ['grpo', 'sapo']:
            return ((per_token_obj * completion_mask_f).sum(-1) / denom_weights_f.sum(-1).clamp(min=1.0)).mean()
        if loss_type in ['bnpo', 'dr_grpo', 'cispo', 'dapo']:
            return (per_token_obj * completion_mask_f).sum() / denom_weights_f.sum().clamp(min=1.0)
        raise ValueError(f'Unknown loss type: {loss_type}')

    @staticmethod
    def _reduce_token_mean(token_values: torch.Tensor,
                           completion_mask: torch.Tensor,
                           loss_type: str,
                           gate_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reduce a per-token diagnostic with the same sequence/token normalization as MAPO loss."""
        completion_mask_f = completion_mask.float()
        token_values_f = torch.nan_to_num(token_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if token_values_f.shape != completion_mask.shape:
            token_values_f = torch.zeros_like(completion_mask_f)
        if gate_weights is None:
            return MegatronMAPOTrainer._reduce_token_objective(token_values_f, completion_mask, loss_type)

        gate_weights_f = torch.nan_to_num(gate_weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if gate_weights_f.shape != completion_mask.shape:
            gate_weights_f = completion_mask_f
        else:
            gate_weights_f = gate_weights_f * completion_mask_f
        return MegatronMAPOTrainer._reduce_token_objective(
            token_values_f * gate_weights_f, completion_mask, loss_type, denom_weights=gate_weights_f)

    def _build_mapo_relevance_weights(self,
                                      delta_h_abs: torch.Tensor,
                                      completion_mask: torch.Tensor,
                                      pos_gate: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build omega/nu relevance weights from detached absolute entropy delta."""
        completion_mask = completion_mask.bool()
        completion_mask_f = completion_mask.float()
        omega_raw = torch.nan_to_num(delta_h_abs.float().abs(), nan=0.0, posinf=0.0, neginf=0.0).detach()
        tau_t = self._build_length_scaled_temperature(completion_mask, self.mapo_mask_temperature)
        # Optionally use a separate (sharper) base temperature for the ν̃ branch.
        _nu_base = self.mapo_nu_temperature if self.mapo_nu_temperature > 0 else self.mapo_mask_temperature
        tau_t_nu = self._build_length_scaled_temperature(completion_mask, _nu_base) if _nu_base != self.mapo_mask_temperature else tau_t
        if pos_gate is None:
            pos_gate_f = completion_mask_f
        else:
            pos_gate_f = torch.nan_to_num(pos_gate.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if pos_gate_f.shape != completion_mask.shape:
                pos_gate_f = completion_mask_f
            else:
                pos_gate_f = pos_gate_f.clamp(min=0.0, max=1.0) * completion_mask_f
        omega_probs = self._masked_temperature_softmax(omega_raw, completion_mask, temperature=tau_t)
        nu_probs = self._masked_temperature_softmax(-omega_raw, completion_mask, temperature=tau_t_nu, gate_mask=pos_gate_f)
        token_count = completion_mask_f.sum(-1, keepdim=True).clamp(min=1.0)
        pos_token_count = pos_gate_f.sum(-1, keepdim=True).clamp(min=1.0)
        omega_tilde = omega_probs * token_count
        nu_tilde = nu_probs * pos_token_count
        return omega_tilde, nu_tilde, omega_raw

    def _build_entropy_filter_mask(
        self,
        delta_h_abs: torch.Tensor,
        text_ref_entropy: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep-mask for the entropy filter: 1.0=keep, 0.0=filtered.

        Token t is filtered from the attention loss iff BOTH:
          - |δh[t]| strictly below the p-th percentile of |δh| in the sequence
          - H_text_ref[t] strictly below the p-th percentile of H_text_ref in the sequence

        Both percentiles are computed over all completion tokens (not just POS-gated tokens)
        to avoid double-applying the POS gate. The joint condition is conservative: it only
        removes near-deterministic, audio-agnostic "template" tokens that satisfy both
        criteria simultaneously.

        When mapo_entropy_filter_percentile == 0.0 the mask is all-ones (feature off).
        """
        p = self.mapo_entropy_filter_percentile
        if p <= 0.0:
            return completion_mask.float()

        q = p / 100.0
        keep_mask = torch.ones_like(completion_mask, dtype=torch.float32)

        for b in range(completion_mask.shape[0]):
            indices = completion_mask[b].nonzero(as_tuple=True)[0]
            n = int(indices.numel())
            if n < 2:
                # Too short for a meaningful percentile; keep all tokens.
                continue

            delta_vals = delta_h_abs[b, indices].float()
            text_h_vals = text_ref_entropy[b, indices].float()

            delta_thresh = torch.quantile(delta_vals, q)
            text_h_thresh = torch.quantile(text_h_vals, q)

            # Strict < avoids over-filtering tied values at the boundary.
            low_delta = delta_h_abs[b] < delta_thresh
            low_text_h = text_ref_entropy[b] < text_h_thresh
            # Guard against non-completion padding positions (e.g. delta_h_abs==0)
            # that could spuriously satisfy low_delta when delta_thresh > 0.
            filter_here = low_delta & low_text_h & completion_mask[b]
            keep_mask[b][filter_here] = 0.0

        return keep_mask

    def _build_temporal_weights(self, completion_mask: torch.Tensor) -> torch.Tensor:
        """Build relative temporal weighting (t/T)^kappa over completion tokens."""
        completion_mask_f = completion_mask.float()
        token_positions = torch.cumsum(completion_mask_f, dim=-1)
        token_count = completion_mask_f.sum(-1, keepdim=True).clamp(min=1.0)
        temporal = torch.pow((token_positions / token_count).clamp(min=0.0, max=1.0), self.mapo_temporal_kappa)
        return temporal * completion_mask_f

    @staticmethod
    def _normalize_mapo_pos_gate(pos_gate: Any, completion_mask: torch.Tensor,
                                 per_token_logps: torch.Tensor) -> torch.Tensor:
        """Return a safe [B, T] POS gate in [0, 1], masked to completion tokens.

        Missing/invalid/misaligned inputs fall back to all-ones (no POS filtering),
        which is the conservative choice to avoid silently dropping learning signal.
        """
        if pos_gate is None:
            pos_gate = torch.ones_like(per_token_logps, dtype=torch.float32, device=per_token_logps.device)
        else:
            pos_gate = torch.nan_to_num(pos_gate.float(), nan=1.0, posinf=1.0, neginf=0.0).to(device=per_token_logps.device)
            if pos_gate.shape != completion_mask.shape:
                pos_gate = torch.ones_like(per_token_logps, dtype=torch.float32, device=per_token_logps.device)
        return pos_gate.clamp(min=0.0, max=1.0) * completion_mask.float()

    @staticmethod
    def _normalize_mapo_task_failed(task_failed: Any, completion_mask: torch.Tensor,
                                    per_token_logps: torch.Tensor) -> torch.Tensor:
        """Return a safe [B] failure gate in [0, 1] for the attention-loss branch.

        Missing/invalid/misaligned inputs fall back to zeros (treat as non-failed),
        preventing accidental over-penalization.
        """
        batch_size = completion_mask.shape[0]
        if task_failed is None:
            task_failed = torch.zeros((batch_size, ), dtype=torch.float32, device=per_token_logps.device)
        else:
            task_failed = torch.nan_to_num(
                task_failed.float(), nan=1.0, posinf=1.0, neginf=0.0).to(device=per_token_logps.device)
            if task_failed.shape[0] != batch_size:
                task_failed = torch.zeros((batch_size, ), dtype=torch.float32, device=per_token_logps.device)
        return task_failed.clamp(min=0.0, max=1.0)

    def _precompute_mapo_weight_cache(self, data: Dict[str, Any]) -> None:
        """Precompute MAPO weights/gates and cache them on the training batch."""
        completion_mask = data.get('completion_mask')
        per_token_logps = data.get('per_token_logps')
        if not isinstance(completion_mask, torch.Tensor) or not isinstance(per_token_logps, torch.Tensor):
            return
        if completion_mask.ndim != 2 or per_token_logps.ndim != 2:
            return

        completion_mask = completion_mask.bool()
        completion_mask_f = completion_mask.float()
        # Use frozen reference model entropies only (H(text_ref) - H(π_ref)) so that nu_tilde
        # is a static target that never shifts as the live policy changes during training.
        ref_per_token_entropy = data.get('ref_per_token_entropy')
        text_ref_per_token_entropy = data.get('text_ref_per_token_entropy')
        if isinstance(text_ref_per_token_entropy, torch.Tensor) and isinstance(ref_per_token_entropy, torch.Tensor):
            delta_h = torch.nan_to_num(
                (text_ref_per_token_entropy - ref_per_token_entropy).float(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            delta_h = torch.zeros_like(per_token_logps, dtype=torch.float32)
        delta_h = delta_h * completion_mask_f

        pos_gate = self._normalize_mapo_pos_gate(data.get('mapo_pos_gate'), completion_mask, per_token_logps)
        omega_tilde, nu_tilde, omega_raw = self._build_mapo_relevance_weights(
            delta_h, completion_mask, pos_gate=pos_gate)
        temporal_weights = self._build_temporal_weights(completion_mask)
        task_failed = self._normalize_mapo_task_failed(data.get('mapo_task_failed'), completion_mask, per_token_logps)

        # Entropy filter: zero nu_tilde for tokens that are jointly low-|δh| AND low-H_text_ref.
        # This removes near-deterministic, audio-agnostic "template" tokens (e.g., "Okay, let's
        # start with ...") from both the KL shape penalty (target_dist) and the mass term
        # (pos_completion_mask) inside _compute_mapo_attention_objective — without touching
        # omega_tilde or the policy-gradient branch.
        if isinstance(text_ref_per_token_entropy, torch.Tensor):
            _text_h = torch.nan_to_num(
                text_ref_per_token_entropy.float(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            _text_h = torch.zeros_like(delta_h)
        entropy_filter_mask = self._build_entropy_filter_mask(
            delta_h.abs(), _text_h, completion_mask)
        nu_tilde = nu_tilde * entropy_filter_mask

        advantages = data.get('advantages')
        if isinstance(advantages, torch.Tensor):
            advantages_abs = torch.nan_to_num(
                advantages.float().abs(), nan=0.0, posinf=0.0, neginf=0.0).to(device=per_token_logps.device)
        else:
            advantages_abs = torch.zeros((completion_mask.shape[0], ), dtype=torch.float32, device=per_token_logps.device)

        data['mapo_cached_omega_tilde'] = omega_tilde
        data['mapo_cached_nu_tilde'] = nu_tilde
        data['mapo_cached_omega_raw'] = omega_raw
        data['mapo_cached_temporal_weights'] = temporal_weights
        data['mapo_cached_pos_gate'] = pos_gate
        data['mapo_cached_task_failed'] = task_failed
        data['mapo_cached_advantages_abs'] = advantages_abs
        data['mapo_cached_entropy_filter_mask'] = entropy_filter_mask

    def _collect_mapo_audio_attention_mass(self, data: Dict[str, Any],
                                           completion_mask: torch.Tensor) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Collect per-token audio attention mass and capture diagnostics."""
        if self._mapo_attention_collector is None:
            return None, {}

        audio_token_mask = data.get('mapo_audio_token_mask')
        if not isinstance(audio_token_mask, torch.Tensor) or audio_token_mask.ndim != 2:
            input_ids = data.get('input_ids')
            audio_token_mask = build_audio_token_mask(input_ids, self._mapo_audio_token_ids)
        seq_lengths = data.get('seq_lengths')
        tp_world_size = 1
        pp_world_size = 1
        try:
            tp_world_size = max(int(mpu.get_tensor_model_parallel_world_size()), 1)
        except Exception:
            tp_world_size = 1
        try:
            pp_world_size = max(int(mpu.get_pipeline_model_parallel_world_size()), 1)
        except Exception:
            pp_world_size = 1
        try:
            audio_mass, diagnostics = self._mapo_attention_collector.compute_audio_attention_mass(
                audio_token_mask=audio_token_mask,
                completion_mask=completion_mask,
                seq_lengths=seq_lengths,
                tp_world_size=tp_world_size)
        except Exception as exc:
            logger.warning(f'MAPO attention mass collection failed: {exc}')
            return None, {}
        diagnostics['mapo_attn_pp_world_size'] = float(pp_world_size)
        target_modules = getattr(self, '_mapo_target_attention_modules', [])
        if target_modules:
            total_core_calls = sum(int(getattr(m, '_mapo_core_forward_calls', 0)) for m in target_modules)
            total_core_success = sum(int(getattr(m, '_mapo_core_forward_success', 0)) for m in target_modules)
            has_tensor_count = sum(
                1 for m in target_modules
                if isinstance(getattr(m, '_mapo_last_attention_probs', None), torch.Tensor))
            diagnostics['mapo_attn_target_modules'] = float(len(target_modules))
            diagnostics['mapo_attn_core_calls'] = float(total_core_calls)
            diagnostics['mapo_attn_core_success'] = float(total_core_success)
            diagnostics['mapo_attn_core_has_tensor'] = float(has_tensor_count)
        diagnostics['mapo_audio_mask_available'] = float(
            isinstance(audio_token_mask, torch.Tensor) and audio_token_mask.ndim == 2)
        if isinstance(audio_token_mask, torch.Tensor) and audio_token_mask.ndim == 2:
            try:
                diagnostics['mapo_audio_mask_on_frac'] = float(
                    audio_token_mask.float().mean().item())
            except Exception:
                pass
        if audio_mass is not None:
            audio_mass = torch.nan_to_num(audio_mass.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return audio_mass, diagnostics

    def _prepare_mapo_audio_mass_for_loss(self, data: Dict[str, Any]) -> None:
        """Precompute and attach audio-mass tensors needed by MAPO loss."""
        if self._mapo_attention_collector is None:
            return
        completion_mask = data.get('completion_mask')
        if not isinstance(completion_mask, torch.Tensor) or completion_mask.ndim != 2:
            return
        audio_mass, diagnostics = self._collect_mapo_audio_attention_mass(data, completion_mask=completion_mask.bool())
        if audio_mass is None and not bool(getattr(self, '_mapo_missing_audio_mass_logged_once', False)):
            target_modules = getattr(self, '_mapo_target_attention_modules', [])
            if target_modules:
                target_module = target_modules[0]
                logger.warning(
                    'MAPO audio mass missing on first occurrence: module=%s core_calls=%s core_success=%s '
                    'last_error=%s q_shape=%s k_shape=%s mask_shape=%s',
                    getattr(target_module, '_mapo_name', target_module.__class__.__name__),
                    int(getattr(target_module, '_mapo_core_forward_calls', 0)),
                    int(getattr(target_module, '_mapo_core_forward_success', 0)),
                    getattr(target_module, '_mapo_last_attention_error', ''),
                    getattr(target_module, '_mapo_core_last_query_shape', None),
                    getattr(target_module, '_mapo_core_last_key_shape', None),
                    getattr(target_module, '_mapo_core_last_mask_shape', None))
            else:
                logger.warning('MAPO audio mass missing on first occurrence: no target attention module tracked.')
            self._mapo_missing_audio_mass_logged_once = True
        if audio_mass is not None:
            data['mapo_audio_mass'] = audio_mass
        data['mapo_attn_diagnostics'] = diagnostics

    def _compute_mapo_attention_objective(self, audio_mass: torch.Tensor, completion_mask: torch.Tensor,
                                          nu_tilde: torch.Tensor, task_failed: torch.Tensor,
                                          advantages: Optional[torch.Tensor] = None,
                                          advantages_abs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MAPO attention loss as KL shape penalty + scalar mass penalty.

        The loss has two orthogonal components:
          - Shape term: KL(nu_tilde_target || actual_audio_dist) — forces the distribution of
            audio attention to match the frozen nu_tilde target. Resists attention inflation because
            uniform mass maximises entropy of actual_audio_dist, diverging from the peaked target.
          - Mass term: -log(total_audio_mass) — shape-agnostic floor that keeps total audio
            attention non-trivial. Weighted by `mapo_mass_lambda`.

        Both terms are gated by task failure and scaled by advantage magnitude.

        `advantages_abs` should be a non-negative per-sequence scale tensor. Callers can
        pass raw `abs(advantages)` or a transformed variant (for example with a floor)
        as long as it remains non-negative.
        """
        floor = getattr(self, 'mapo_task_fail_gate_floor', 0.0)
        mass_lambda = getattr(self, 'mapo_mass_lambda', 0.1)
        task_fail_gate = task_failed.float().clamp(min=floor, max=1.0)  # [B]
        if advantages_abs is not None:
            advantage_scale = advantages_abs.float().clamp(min=0.0)  # [B]
        elif advantages is not None:
            advantage_scale = advantages.float().abs()  # [B]
        else:
            raise ValueError('Either advantages or advantages_abs must be provided for MAPO attention objective.')

        completion_mask_f = completion_mask.float()

        # --- Shape term: KL(target || actual) ---
        # Target: nu_tilde is already POS-gated and sequence-normalised; re-normalise to a
        # proper probability distribution over completion tokens (detached — no gradient path).
        target_dist = nu_tilde.detach().float() * completion_mask_f
        target_dist = target_dist / target_dist.sum(-1, keepdim=True).clamp(min=1e-8)  # [B, T]

        # Actual: normalise audio_mass over completion tokens (gradients flow through this).
        actual_dist = audio_mass.float() * completion_mask_f
        actual_dist = actual_dist / actual_dist.sum(-1, keepdim=True).clamp(min=1e-8)  # [B, T]

        # One-sided forward KL: only penalise under-attention (ā_t < ν̄_t).
        # Positions where attention already meets or exceeds the target contribute
        # zero loss and zero gradient, preserving natural attention peaks.
        # Gradient: ∂L/∂ā_t = -ν̄_t / ā_t when ā_t < ν̄_t, else 0.
        log_ratio = (
            torch.log(target_dist.clamp(min=_MAPO_ATTN_LOG_EPS))
            - torch.log(actual_dist.clamp(min=_MAPO_ATTN_LOG_EPS))
        ).clamp(min=0.0)  # [B, T]  zero where actual ≥ target

        kl_per_token = target_dist * log_ratio  # [B, T]
        per_seq_kl = (kl_per_token * completion_mask_f).sum(-1)  # [B]

        # --- Mass term: -log(total_audio_mass over POS-gated completion tokens) ---
        # Restrict to the same token set as the KL target: positions where nu_tilde > 0
        # (i.e. POS-gated AND in completion). This prevents the model from satisfying the
        # mass penalty by inflating audio attention on non-substantive tokens (prepositions,
        # punctuation, etc.) that carry zero weight in the KL and are irrelevant to the
        # modality grounding objective.
        pos_completion_mask = (target_dist > 0).float()  # derived from nu_tilde; no new args
        total_mass = (audio_mass.float() * pos_completion_mask).sum(-1)  # [B]
        per_seq_mass = -torch.log(total_mass.clamp(min=_MAPO_ATTN_LOG_EPS))  # [B]

        # --- Combined gated loss ---
        gate = task_fail_gate * advantage_scale  # [B]
        per_seq_attn_obj = gate * (per_seq_kl + mass_lambda * per_seq_mass)  # [B]
        attn_loss = per_seq_attn_obj.mean()

        # Per-token proxy for diagnostics logging (not used in the loss computation).
        per_token_attn_obj = kl_per_token * gate.unsqueeze(-1)  # [B, T]
        return per_token_attn_obj, attn_loss

    def _get_mapo_target_trainable_params(self) -> List[torch.nn.Parameter]:
        modules = getattr(self, '_mapo_target_attention_modules', None)
        if not isinstance(modules, (list, tuple)):
            return []
        params: List[torch.nn.Parameter] = []
        seen = set()
        for module in modules:
            if module is None:
                continue
            for param in module.parameters(recurse=True):
                if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(param)
        return params

    def _should_run_mapo_attn_grad_probe(self) -> bool:
        if not getattr(self, 'mapo_debug_attn_grad_probe', False):
            return False
        interval = int(getattr(self, 'mapo_debug_attn_grad_probe_interval', 0) or 0)
        if interval <= 0:
            return True
        iteration = int(getattr(self.args, 'curr_iteration', 0) or 0)
        if iteration <= 0:
            return True
        return (iteration % interval) == 0

    @staticmethod
    def _empty_mapo_attn_probe_report() -> Dict[str, float]:
        return {
            'mapo/attn_probe_active': 0.0,
            'mapo/attn_probe_audio_mass_requires_grad': 0.0,
            'mapo/attn_probe_audio_mass_grad_nonzero': 0.0,
            'mapo/attn_probe_audio_mass_grad_norm': 0.0,
            'mapo/attn_probe_audio_mass_grad_abs_mean': 0.0,
            'mapo/attn_probe_param_count': 0.0,
            'mapo/attn_probe_grad_nonzero_count': 0.0,
            'mapo/attn_probe_grad_param_frac': 0.0,
            'mapo/attn_probe_grad_norm': 0.0,
            'mapo/attn_probe_grad_abs_mean': 0.0,
        }

    @staticmethod
    def _mapo_attn_probe_report_to_tensors(report: Optional[Dict[str, float]],
                                           device: torch.device) -> Dict[str, torch.Tensor]:
        if not isinstance(report, dict):
            report = MegatronMAPOTrainer._empty_mapo_attn_probe_report()
        return {
            key: torch.tensor(float(value), dtype=torch.float32, device=device)
            for key, value in report.items()
        }

    def _new_mapo_attn_probe_live_state(self, iteration: int) -> Dict[str, Any]:
        return {
            'iteration': int(iteration),
            'active': 0.0,
            'audio_mass_requires_grad': 0.0,
            'audio_mass_grad_nonzero': 0.0,
            'audio_mass_grad_norm_sq': 0.0,
            'audio_mass_grad_abs_sum': 0.0,
            'audio_mass_grad_elem_count': 0.0,
            'param_count': 0.0,
            'param_nonzero_ids': set(),
            'param_grad_norm_sq': 0.0,
            'param_grad_abs_sum': 0.0,
            'param_grad_elem_count': 0.0,
            'param_hook_handles': [],
            'param_hooks_registered': False,
        }

    @staticmethod
    def _accumulate_mapo_attn_probe_grad(state: Dict[str, Any], grad: torch.Tensor, prefix: str) -> None:
        if not isinstance(grad, torch.Tensor):
            return
        grad_f = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        if grad_f.numel() <= 0:
            return
        state[f'{prefix}_grad_norm_sq'] += float(grad_f.pow(2).sum().item())
        state[f'{prefix}_grad_abs_sum'] += float(grad_f.abs().sum().item())
        state[f'{prefix}_grad_elem_count'] += float(grad_f.numel())

    def _finalize_mapo_attn_probe_live_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, float]:
        report = self._empty_mapo_attn_probe_report()
        if not isinstance(state, dict):
            return report
        for handle in state.get('param_hook_handles', []):
            try:
                handle.remove()
            except Exception:
                pass
        report['mapo/attn_probe_active'] = float(state.get('active', 0.0))
        report['mapo/attn_probe_audio_mass_requires_grad'] = float(state.get('audio_mass_requires_grad', 0.0))
        report['mapo/attn_probe_audio_mass_grad_nonzero'] = float(state.get('audio_mass_grad_nonzero', 0.0))
        report['mapo/attn_probe_audio_mass_grad_norm'] = float(state.get('audio_mass_grad_norm_sq', 0.0))**0.5
        audio_mass_grad_elem_count = float(state.get('audio_mass_grad_elem_count', 0.0))
        if audio_mass_grad_elem_count > 0.0:
            report['mapo/attn_probe_audio_mass_grad_abs_mean'] = (
                float(state.get('audio_mass_grad_abs_sum', 0.0)) / audio_mass_grad_elem_count)
        param_count = float(state.get('param_count', 0.0))
        nonzero_param_count = float(len(state.get('param_nonzero_ids', set())))
        report['mapo/attn_probe_param_count'] = param_count
        report['mapo/attn_probe_grad_nonzero_count'] = nonzero_param_count
        if param_count > 0.0:
            report['mapo/attn_probe_grad_param_frac'] = nonzero_param_count / param_count
        report['mapo/attn_probe_grad_norm'] = float(state.get('param_grad_norm_sq', 0.0))**0.5
        param_grad_elem_count = float(state.get('param_grad_elem_count', 0.0))
        if param_grad_elem_count > 0.0:
            report['mapo/attn_probe_grad_abs_mean'] = (
                float(state.get('param_grad_abs_sum', 0.0)) / param_grad_elem_count)
        return report

    def _current_mapo_attn_probe_state(self) -> Optional[Dict[str, Any]]:
        if not self._should_run_mapo_attn_grad_probe():
            return None
        iteration = int(getattr(self.args, 'curr_iteration', 0) or 0)
        live_state = getattr(self, '_mapo_attn_probe_live_state', None)
        if not isinstance(live_state, dict) or int(live_state.get('iteration', -1)) != iteration:
            live_state = self._new_mapo_attn_probe_live_state(iteration)
            self._mapo_attn_probe_live_state = live_state
        return live_state

    def _register_mapo_attn_grad_hooks(self, audio_mass: Optional[torch.Tensor]) -> None:
        state = self._current_mapo_attn_probe_state()
        if state is None:
            return
        state['active'] = 1.0

        if isinstance(audio_mass, torch.Tensor) and audio_mass.requires_grad:
            state['audio_mass_requires_grad'] = 1.0

            def _audio_mass_hook(grad):
                self._accumulate_mapo_attn_probe_grad(state, grad, prefix='audio_mass')
                grad_f = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if bool((grad_f.abs() > 0).any().item()):
                    state['audio_mass_grad_nonzero'] = 1.0
                return grad

            audio_mass.register_hook(_audio_mass_hook)

        target_params = self._get_mapo_target_trainable_params()
        state['param_count'] = float(len(target_params))
        if target_params and not bool(state.get('param_hooks_registered', False)):
            hook_handles = []
            for param_idx, param in enumerate(target_params):
                def _param_hook(grad, idx=param_idx):
                    self._accumulate_mapo_attn_probe_grad(state, grad, prefix='param')
                    grad_f = torch.nan_to_num(grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    if bool((grad_f.abs() > 0).any().item()):
                        state['param_nonzero_ids'].add(int(idx))
                    return grad

                hook_handles.append(param.register_hook(_param_hook))
            state['param_hook_handles'] = hook_handles
            state['param_hooks_registered'] = True

    @staticmethod
    def _live_mapo_attn_probe_static_metrics(state: Optional[Dict[str, Any]],
                                             device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        if not isinstance(state, dict):
            return {
                'mapo/attn_probe_active': zero,
                'mapo/attn_probe_audio_mass_requires_grad': zero,
                'mapo/attn_probe_param_count': zero,
            }
        return {
            'mapo/attn_probe_active': torch.tensor(float(state.get('active', 0.0)),
                                                   dtype=torch.float32,
                                                   device=device),
            'mapo/attn_probe_audio_mass_requires_grad': torch.tensor(
                float(state.get('audio_mass_requires_grad', 0.0)),
                dtype=torch.float32,
                device=device),
            'mapo/attn_probe_param_count': torch.tensor(float(state.get('param_count', 0.0)),
                                                        dtype=torch.float32,
                                                        device=device),
        }

    def _finalize_mapo_attn_probe_for_logging(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if not getattr(self, 'mapo_debug_attn_grad_probe', False):
            return {}
        report = self._empty_mapo_attn_probe_report()
        live_state = getattr(self, '_mapo_attn_probe_live_state', None)
        if isinstance(live_state, dict):
            report = self._finalize_mapo_attn_probe_live_state(live_state)
            self._mapo_attn_probe_live_state = None
        self._mapo_attn_probe_prev_report = report
        return self._mapo_attn_probe_report_to_tensors(report, device=device)

    def _resolve_failure_reward_index(self, reward_width: int) -> int:
        """Resolve and cache the reward-function index used for failure gating."""
        if self._mapo_failure_reward_idx is not None and self._mapo_failure_reward_idx < reward_width:
            name = self.reward_func_names[self._mapo_failure_reward_idx]
            if name == self.mapo_failure_reward_name:
                return self._mapo_failure_reward_idx
        for idx, name in enumerate(self.reward_func_names):
            if name == self.mapo_failure_reward_name:
                self._mapo_failure_reward_idx = idx
                return idx
        raise ValueError(
            f'MAPO failure reward `{self.mapo_failure_reward_name}` not found in reward_funcs: {self.reward_func_names}')

    def _init_mapo_attention_collector(self):
        """Initialize MAPO single-pass attention collector and layer hook wiring."""
        self._mapo_attention_collector = None
        self._mapo_audio_token_ids = []
        if not self.unwrapped_models:
            return
        pp_world_size = 1
        is_pp_last_stage = True
        try:
            pp_world_size = max(int(mpu.get_pipeline_model_parallel_world_size()), 1)
        except Exception:
            pp_world_size = 1
        try:
            is_pp_last_stage = bool(mpu.is_pipeline_last_stage())
        except Exception:
            is_pp_last_stage = True
        if pp_world_size > 1 and not is_pp_last_stage:
            logger.info(
                'MAPO attention collector is disabled on non-last PP stage '
                '(pipeline_model_parallel_size=%s).',
                pp_world_size)
            return
        try:
            # Parse the flexible layer specification.
            parsed_global_layers = parse_attention_layer_spec(str(self.mapo_attention_layers))

            # Discover local attention layers on this PP stage.
            local_attention_layers: List[int] = []
            debug_attention_like_names: List[str] = []
            for module_name, module in self.unwrapped_models[0].named_modules():
                module_name_l = str(module_name).lower()
                if ('attn' in module_name_l or 'attention' in module_name_l) and len(debug_attention_like_names) < 40:
                    debug_attention_like_names.append(str(module_name))
                if not self._is_qwen_thinker_text_attention(module_name, module):
                    continue
                layer_idx = self._parse_layer_index(module_name)
                if layer_idx is not None:
                    local_attention_layers.append(layer_idx)
            local_attention_layers = sorted(set(local_attention_layers))
            if not local_attention_layers:
                raise ValueError(
                    'No local attention layers matched MAPO layer parser on current PP stage. '
                    f'Attention-like module name samples: {debug_attention_like_names}')

            # Map global layer indices to local layer indices.
            local_layer_count = len(local_attention_layers)
            local_max_layer = max(local_attention_layers)
            local_min_layer = min(local_attention_layers)

            if parsed_global_layers is None:
                # "all" — use all local layers on this stage.
                target_local_layers = set(local_attention_layers)
            else:
                # Map each global layer index to a local index.
                # Global layer offset: e.g. if local layers are 0..23 and PP=2,
                # the global offset for the last stage is total_layers - local_count.
                # We infer the offset from the local layer numbering.
                global_offset = local_min_layer  # local layers usually start at 0 on last stage
                target_local_layers: Set[int] = set()
                for global_idx in parsed_global_layers:
                    # Try direct match first (for PP=1 or when global==local).
                    if global_idx in local_attention_layers:
                        target_local_layers.add(global_idx)
                    else:
                        # Try offset-based mapping: global_idx maps to local if it's within
                        # the range [global_offset_of_this_stage, global_offset + local_count).
                        # For PP last stage, local modules are named 0..N-1.
                        # If user provides global indices like 45,46,47 with 48 total layers
                        # and PP=2 (24 per stage), the last stage has local 0..23,
                        # and global 45 -> local 45 - 24 = 21.
                        # We need to estimate the global-to-local offset.
                        estimated_local = global_idx - (local_max_layer + 1 - local_layer_count)
                        # Clamp: keep only if estimated local index exists.
                        if estimated_local in local_attention_layers:
                            target_local_layers.add(estimated_local)

                if not target_local_layers:
                    # Fallback: if no mapping worked (e.g. naming scheme mismatch),
                    # select the last N local layers matching the requested count.
                    n_requested = len(parsed_global_layers)
                    target_local_layers = set(local_attention_layers[-n_requested:])
                    logger.warning(
                        'MAPO global-to-local layer mapping produced no matches for spec=%s. '
                        'Falling back to last %s local layer(s): %s.',
                        self.mapo_attention_layers, n_requested, sorted(target_local_layers))

            collector = MAPOAttentionCollector(
                target_local_layers,
                head_reduce=self.mapo_attention_head_reduce,
                layer_reduce=self.mapo_attention_layer_reduce)
            collector.attach(self.unwrapped_models[0])
            self._mapo_attention_collector = collector

            eager_override_modules = self._configure_mapo_local_eager_override(
                self.unwrapped_models[0], target_layers=target_local_layers)
            if eager_override_modules <= 0:
                raise ValueError(
                    f'Failed to apply MAPO eager override on local layers {sorted(target_local_layers)}; '
                    'no target attention modules found.')
            override_debug = getattr(self, '_mapo_eager_override_debug', [])
            if override_debug:
                logger.info('MAPO eager override target modules (first %s): %s',
                            len(override_debug), override_debug)
            else:
                logger.warning('MAPO eager override debug list is empty despite nonzero override count.')

            self._mapo_audio_token_ids = detect_audio_token_ids(self.unwrapped_models[0], tokenizer=self.processing_class)
            if not self._mapo_audio_token_ids:
                logger.warning('MAPO could not detect explicit audio token ids; attention mass may stay near zero.')
            if pp_world_size > 1:
                logger.info(
                    'MAPO attention collector is enabled on PP last stage '
                    '(pipeline_model_parallel_size=%s).',
                    pp_world_size)
            logger.info(
                'MAPO attention is configured (layer_spec=%s, target_local_layers=%s, '
                'local_layers=%s, eager_override_modules=%s, head_reduce=%s, layer_reduce=%s).',
                self.mapo_attention_layers, sorted(target_local_layers), local_attention_layers,
                eager_override_modules, self.mapo_attention_head_reduce, self.mapo_attention_layer_reduce)
        except Exception as exc:
            self._mapo_attention_collector = None
            self._mapo_audio_token_ids = []
            raise RuntimeError(f'Failed to initialize MAPO attention collector: {exc}') from exc

    def _init_grpo_params(self):
        """Initialize MAPO runtime parameters and enable phase-2 entropy-weighted PG path."""
        super()._init_grpo_params()
        args = self.args
        self.eta = args.eta
        self.mapo_advantage_floor_eps = max(float(getattr(args, 'mapo_advantage_floor_eps', 0.0)), 0.0)
        self.mapo_task_fail_gate_floor = float(getattr(args, 'mapo_task_fail_gate_floor', 0.0))
        self._mapo_attention_loss_enabled = bool(abs(float(self.eta)) > 0.0)
        self.mapo_mask_temperature = args.mapo_mask_temperature
        self.mapo_temporal_kappa = args.mapo_temporal_kappa
        self.mapo_attention_layers = args.mapo_attention_layers
        self.mapo_attention_head_reduce = getattr(args, 'mapo_attention_head_reduce', 'max')
        self.mapo_attention_layer_reduce = getattr(args, 'mapo_attention_layer_reduce', 'mean')
        self.mapo_failure_reward_name = args.mapo_failure_reward_name
        self.mapo_failure_threshold = args.mapo_failure_threshold
        self.mapo_pos_tags = self._parse_mapo_pos_tags(args.mapo_pos_tags)
        self.mapo_attention_only = bool(getattr(args, 'mapo_attention_only', False))
        self.mapo_debug_attn_grad_probe = bool(getattr(args, 'mapo_debug_attn_grad_probe', False))
        self.mapo_debug_attn_grad_probe_interval = int(
            getattr(args, 'mapo_debug_attn_grad_probe_interval', 0) or 0)
        self.text_only_modality_scope = args.text_only_modality_scope
        self.mapo_entropy_filter_percentile = float(
            getattr(args, 'mapo_entropy_filter_percentile', 0.0))
        self.mapo_nu_temperature = float(getattr(args, 'mapo_nu_temperature', 0.2))
        self._mapo_failure_reward_idx = None

        # MAPO phase-2 always requires policy entropy for Delta-H weighting.
        self.top_entropy_quantile = 1.0
        self.compute_entropy = True
        if not self._mapo_attention_loss_enabled:
            logger.info(
                'MAPO attention loss branch is disabled because eta=%s. '
                'MAPO entropy-weighted PG branch remains enabled.',
                self.eta)
        if self.mapo_attention_only:
            logger.warning(
                'MAPO attention-only diagnostic mode is enabled: policy-gradient and KL losses '
                'will be excluded from backprop; only eta * attn_loss will be optimized.')
        if self.mapo_debug_attn_grad_probe:
            logger.warning(
                'MAPO attention gradient probe is enabled: the trainer will register backward hooks '
                'on MAPO audio_mass and target-layer trainable params; reported metrics lag by one iteration.')
        self.text_ref_models = []
        self._mapo_attention_collector = None
        self._mapo_audio_token_ids = []
        self._mapo_attn_probe_live_state = None
        self._mapo_attn_probe_prev_report = None
        self._warn_deprecated_mapo_args()

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        """Build policy/reference models and initialize MAPO collector after setup."""
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
        result = super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)
        if self._mapo_attention_loss_enabled:
            self._init_mapo_attention_collector()
        return result

    @contextmanager
    def _text_ref_context(self):
        """Yield a context where the text-reference model is used without training adapters."""
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

    def forward_step(self, data_iterator, model):
        """Run one MAPO forward step and precompute per-step MAPO caches."""
        if self._mapo_attention_loss_enabled and self._mapo_attention_collector is not None:
            self._mapo_attention_collector.reset()
        output_tensor, loss_func = super().forward_step(data_iterator, model)
        try:
            data = getattr(loss_func, 'keywords', {}).get('data')
            if isinstance(data, dict):
                self._precompute_mapo_weight_cache(data)
                if self._mapo_attention_loss_enabled:
                    self._prepare_mapo_audio_mass_for_loss(data)
        except Exception as exc:
            logger.warning(f'MAPO forward-step attention mass precompute failed: {exc}')
        return output_tensor, loss_func

    def _score_completions(self, inputs):
        """Annotate rollouts with task-failure and POS-gate metadata for MAPO."""
        if not self._mapo_attention_loss_enabled:
            return super()._score_completions(inputs)
        rewards_per_func = super()._score_completions(inputs)
        failure_idx = self._resolve_failure_reward_index(rewards_per_func.shape[1])
        for idx, sample in enumerate(inputs):
            reward_value = rewards_per_func[idx, failure_idx]
            reward_scalar = float(reward_value.item())
            task_failed = 1.0 if (not torch.isfinite(reward_value).item() or reward_scalar <= self.mapo_failure_threshold) else 0.0
            sample['mapo_task_failed'] = task_failed

            token_ids = self._flatten_response_token_ids(sample.get('response_token_ids'))
            pos_gate, fallback_used = build_pos_token_gate(
                token_ids=token_ids, tokenizer=self.processing_class, target_pos_tags=self.mapo_pos_tags)
            sample['mapo_pos_token_gate'] = pos_gate
            sample['mapo_pos_fallback'] = bool(fallback_used)
        return rewards_per_func

    def _prepare_model_inputs(self, inputs):
        """Remove MAPO-only cached tensors before policy forward invocation."""
        model_inputs = super()._prepare_model_inputs(inputs)
        for key in [
                'text_ref_per_token_logps', 'text_ref_per_token_entropy', 'ref_per_token_entropy',
                'mapo_pos_gate', 'mapo_task_failed',
                'mapo_pos_fallback', 'mapo_audio_token_mask', 'mapo_audio_mass', 'mapo_attn_diagnostics',
                'mapo_cached_omega_tilde', 'mapo_cached_nu_tilde', 'mapo_cached_omega_raw', 'mapo_cached_temporal_weights',
                'mapo_cached_pos_gate', 'mapo_cached_task_failed', 'mapo_cached_advantages_abs',
                'mapo_cached_entropy_filter_mask'
        ]:
            model_inputs.pop(key, None)
        return model_inputs

    def _get_encoded_batch(self, encoded_list, rollout_batch, template):
        """Project sample-level MAPO annotations into aligned batch tensors."""
        if not self._mapo_attention_loss_enabled:
            return super()._get_encoded_batch(encoded_list, rollout_batch, template)
        encoded_batch = super()._get_encoded_batch(encoded_list, rollout_batch, template)
        completion_mask = encoded_batch['completion_mask'].bool()
        batch_size = completion_mask.shape[0]
        seq_len = completion_mask.shape[1]

        mapo_pos_gate = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=self.device)
        mapo_task_failed = torch.zeros((batch_size, ), dtype=torch.float32, device=self.device)
        mapo_pos_fallback = torch.zeros((batch_size, ), dtype=torch.float32, device=self.device)

        for idx in range(batch_size):
            sample = rollout_batch[idx] if idx < len(rollout_batch) else {}
            completion_indices = completion_mask[idx].nonzero(as_tuple=True)[0]
            token_count = int(completion_indices.numel())
            gate_values, fallback_used = self._align_pos_gate_values(
                sample.get('mapo_pos_token_gate'), token_count=token_count, device=self.device)
            if token_count > 0:
                mapo_pos_gate[idx, completion_indices] = gate_values

            task_failed = sample.get('mapo_task_failed', 0.0)
            mapo_task_failed[idx] = float(task_failed)

            fallback_flag = bool(sample.get('mapo_pos_fallback', False)) or fallback_used
            mapo_pos_fallback[idx] = 1.0 if fallback_flag else 0.0

        mapo_audio_token_mask = self._build_aligned_audio_token_mask(
            encoded_batch=encoded_batch, completion_mask=completion_mask)
        if not isinstance(mapo_audio_token_mask, torch.Tensor):
            mapo_audio_token_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)

        encoded_batch['mapo_pos_gate'] = mapo_pos_gate
        encoded_batch['mapo_task_failed'] = mapo_task_failed
        encoded_batch['mapo_pos_fallback'] = mapo_pos_fallback
        encoded_batch['mapo_audio_token_mask'] = mapo_audio_token_mask
        return encoded_batch

    def _build_text_only_inputs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop modality inputs to build text-only reference forward inputs."""
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
        """Compute per-token log-probs and entropy for the text-reference model."""
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
        """Populate reference and text-reference log-prob/entropy tensors in the training batch.

        Fully overrides the parent to avoid a redundant forward pass: the parent's
        null_ref_context forward only returns logps (discards entropy). Here we call
        _compute_per_token_logps_and_entropy() for the frozen GRPO reference model so we
        get both ref_per_token_logps and ref_per_token_entropy in a single pass, which is
        needed to build the frozen nu_tilde target for the KL attention loss.
        """
        seq_lengths = batch['seq_lengths']
        batch_size = batch['num_samples']
        max_seq_len = batch['completion_mask'].shape[1]
        model_inputs = self._prepare_model_inputs(batch)

        # 1) Frozen GRPO reference model: logps + entropy in one forward pass.
        #    Conditioned on: beta != 0 (KL reg needs logps) OR attention loss (needs entropy).
        if self.beta != 0.0 or self._mapo_attention_loss_enabled:
            with torch.no_grad(), self.null_ref_context() as ref_models:
                assert len(ref_models) == 1, 'MAPO currently does not support VPP.'
                ref_model = ref_models[0]
                ref_per_token_logps, ref_per_token_entropy = self._compute_per_token_logps_and_entropy(
                    ref_model, deepcopy(model_inputs),
                    batch_size=batch_size, max_seq_len=max_seq_len, seq_lengths=seq_lengths)
                batch['ref_per_token_logps'] = ref_per_token_logps
                batch['ref_per_token_entropy'] = ref_per_token_entropy

        # 2) Old policy logps (no entropy needed — same as parent behaviour).
        old_per_token_logps_raw = self.model_forward(
            self.unwrapped_models[0], iter([deepcopy(model_inputs)]), no_grad=True, per_token=True)['logps']
        if self.template.padding_free:
            old_per_token_logps, _ = pad_logps_back_to_batch(
                logps_rmpad=old_per_token_logps_raw,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths)
        else:
            old_per_token_logps = old_per_token_logps_raw
        batch['old_per_token_logps'] = old_per_token_logps

        # 3) Text-only reference: logps + entropy (existing MAPO logic, unchanged).
        text_only_inputs = self._build_text_only_inputs(model_inputs)
        with torch.no_grad(), self._text_ref_context() as text_ref_models:
            assert len(text_ref_models) == 1, 'MAPO currently does not support VPP.'
            text_ref_model = text_ref_models[0]
            text_ref_per_token_logps, text_ref_per_token_entropy = self._compute_per_token_logps_and_entropy(
                text_ref_model, text_only_inputs,
                batch_size=batch_size, max_seq_len=max_seq_len, seq_lengths=seq_lengths)
            batch['text_ref_per_token_logps'] = text_ref_per_token_logps
            batch['text_ref_per_token_entropy'] = text_ref_per_token_entropy

        return batch

    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
        """Assemble MAPO PG/KL/attention losses and emit training diagnostics."""
        args = get_args()
        advantages = data['advantages']
        completion_mask = data['completion_mask'].bool()
        packed_seq_params = data.get('packed_seq_params')
        truncated_mask = data['truncated_mask']
        seq_lengths = data['seq_lengths']
        micro_batch_size = self.micro_batch_size

        per_token_logps = data.get('per_token_logps')
        per_token_entropy = data.get('per_token_entropy')
        ref_per_token_logps = data.get('ref_per_token_logps')
        old_per_token_logps = data.get('old_per_token_logps')
        rollout_per_token_logps = data.get('rollout_per_token_logps')
        text_ref_per_token_entropy = data.get('text_ref_per_token_entropy')

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

        if (not self.mapo_attention_only) and self.beta != 0.0 and ref_per_token_logps is not None and not self.kl_in_reward:
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

        if self.off_policy_sequence_mask_delta is not None:
            old_policy_per_token_logps = rollout_per_token_logps if rollout_per_token_logps is not None \
                else old_per_token_logps
            off_policy_seq_mask = self._compute_off_policy_sequence_mask(per_token_logps, old_policy_per_token_logps,
                                                                         completion_mask, advantages)
            off_policy_seq_mask_expanded = off_policy_seq_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & off_policy_seq_mask_expanded

        completion_mask_f = completion_mask.float()
        self._precompute_mapo_weight_cache(data)

        omega_tilde = data.get('mapo_cached_omega_tilde')
        nu_tilde = data.get('mapo_cached_nu_tilde')
        omega_raw = data.get('mapo_cached_omega_raw')
        temporal_weights = data.get('mapo_cached_temporal_weights')
        pos_gate = data.get('mapo_cached_pos_gate')
        task_failed = data.get('mapo_cached_task_failed')
        advantages_abs = data.get('mapo_cached_advantages_abs')

        if not isinstance(omega_tilde, torch.Tensor) or omega_tilde.shape != completion_mask.shape:
            ref_per_token_entropy_fb = data.get('ref_per_token_entropy')
            if text_ref_per_token_entropy is not None and ref_per_token_entropy_fb is not None:
                delta_h = torch.nan_to_num(
                    (text_ref_per_token_entropy - ref_per_token_entropy_fb).float(), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                delta_h = torch.zeros_like(per_token_logps, dtype=torch.float32)
            delta_h = delta_h * completion_mask_f
            pos_gate = self._normalize_mapo_pos_gate(data.get('mapo_pos_gate'), completion_mask, per_token_logps)
            omega_tilde, nu_tilde, omega_raw = self._build_mapo_relevance_weights(
                delta_h, completion_mask, pos_gate=pos_gate)
            temporal_weights = self._build_temporal_weights(completion_mask)
            task_failed = self._normalize_mapo_task_failed(data.get('mapo_task_failed'), completion_mask, per_token_logps)
            advantages_abs = torch.nan_to_num(
                advantages.float().abs(), nan=0.0, posinf=0.0, neginf=0.0).to(device=per_token_logps.device)

        pg_weights = completion_mask_f * omega_tilde
        per_token_pg_obj = per_token_pg_loss

        if self.loss_type in ['grpo', 'sapo']:
            pg_loss = ((per_token_pg_obj * pg_weights).sum(-1) / pg_weights.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type in ['bnpo', 'dr_grpo', 'cispo', 'dapo']:
            pg_loss = (per_token_pg_obj * pg_weights).sum() / pg_weights.sum().clamp(min=1.0)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        reg_loss = None
        if self.mapo_attention_only:
            # Preserve the standard PP backward topology with an exact zero-weight anchor
            # through the normal policy loss graph, while removing non-attention gradients.
            loss = pg_loss * 0.0
        elif self.beta != 0.0 and per_token_kl is not None:
            reg_loss = self._reduce_token_objective(per_token_kl, completion_mask, self.loss_type)
            loss = pg_loss + self.beta * reg_loss
        else:
            loss = pg_loss

        attn_diagnostics = data.get('mapo_attn_diagnostics')
        if not isinstance(attn_diagnostics, dict):
            attn_diagnostics = {}
        pos_gate = self._normalize_mapo_pos_gate(pos_gate, completion_mask, per_token_logps)
        task_failed = self._normalize_mapo_task_failed(task_failed, completion_mask, per_token_logps)
        audio_mass = torch.zeros_like(per_token_logps, dtype=torch.float32, device=per_token_logps.device)
        attn_loss = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        adv_abs_mean = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        adv_scale_mean = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        adv_floor_applied_frac = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        adv_floor_lift_mean = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        effective_task_gate_mean = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        attn_prefactor_mean = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
        attn_grad_probe_metrics: Dict[str, torch.Tensor] = {}
        if self.mapo_debug_attn_grad_probe:
            attn_grad_probe_metrics = self._mapo_attn_probe_report_to_tensors(
                self._empty_mapo_attn_probe_report(), per_token_logps.device)

        if self._mapo_attention_loss_enabled:
            temporal_weights = torch.nan_to_num(
                temporal_weights.float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=per_token_logps.device)
            advantages_abs = torch.nan_to_num(
                advantages_abs.float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=per_token_logps.device)
            adv_abs_mean = advantages_abs.mean()

            # Advantage flooring is intentionally scoped to failed samples so non-failed
            # trajectories preserve the original MAPO attention scale.
            advantage_scale = advantages_abs
            if self.mapo_advantage_floor_eps > 0.0:
                failed_mask = task_failed > 0.0
                floor_values = torch.full_like(advantage_scale, self.mapo_advantage_floor_eps)
                floored_scale = torch.where(failed_mask, torch.maximum(advantage_scale, floor_values), advantage_scale)
                floor_applied_mask = failed_mask & (floored_scale > advantage_scale)
                adv_floor_applied_frac = floor_applied_mask.float().sum() / failed_mask.float().sum().clamp(min=1.0)
                advantage_scale = floored_scale
            adv_scale_mean = advantage_scale.mean()
            adv_floor_lift_mean = (advantage_scale - advantages_abs).mean()
            effective_task_gate = task_failed.float().clamp(min=self.mapo_task_fail_gate_floor, max=1.0)
            effective_task_gate_mean = effective_task_gate.mean()
            attn_prefactor_mean = (effective_task_gate * advantage_scale).mean()

            audio_mass = data.get('mapo_audio_mass')
            if isinstance(audio_mass, torch.Tensor):
                audio_mass = torch.nan_to_num(
                    audio_mass.float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=per_token_logps.device)
            else:
                audio_mass = None

            if audio_mass is None:
                collected_audio_mass, collected_diagnostics = self._collect_mapo_audio_attention_mass(data, completion_mask)
                if isinstance(collected_diagnostics, dict):
                    attn_diagnostics = {**attn_diagnostics, **collected_diagnostics}
                audio_mass = collected_audio_mass
            if audio_mass is None or audio_mass.shape != completion_mask.shape:
                attn_diagnostics['mapo_single_pass_missing_audio_mass'] = 1.0
                audio_mass = torch.zeros_like(per_token_logps, dtype=torch.float32, device=per_token_logps.device)
                attn_loss = torch.zeros((), dtype=torch.float32, device=per_token_logps.device)
            else:
                audio_mass = audio_mass.to(device=per_token_logps.device, dtype=torch.float32)
                _, attn_loss = self._compute_mapo_attention_objective(
                    audio_mass=audio_mass,
                    completion_mask=completion_mask,
                    nu_tilde=nu_tilde,
                    task_failed=task_failed,
                    advantages_abs=advantage_scale)

            self._register_mapo_attn_grad_hooks(audio_mass=audio_mass)
            if self.mapo_debug_attn_grad_probe:
                live_probe_state = getattr(self, '_mapo_attn_probe_live_state', None)
                attn_grad_probe_metrics.update(
                    self._live_mapo_attn_probe_static_metrics(live_probe_state, per_token_logps.device))
            loss = loss + self.eta * attn_loss
        else:
            pos_gate = torch.zeros_like(pos_gate)
            task_failed = torch.zeros_like(task_failed)
            attn_diagnostics['mapo_attn_branch_disabled_by_eta'] = 1.0

        avg_metric = {'loss': loss.clone().detach()}
        total_lengths = gather(lengths, group=mpu.get_data_parallel_group(with_context_parallel=True))

        valid_token_denom = completion_mask_f.sum().clamp(min=1.0)
        mask_weights = torch.nan_to_num(omega_tilde.float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=loss.device)
        mask_weights = mask_weights * completion_mask_f
        masked_min_weights = torch.where(
            completion_mask, mask_weights, torch.full_like(mask_weights, float('inf')))
        mask_weight_min = torch.nan_to_num(
            masked_min_weights.min(), nan=0.0, posinf=0.0, neginf=0.0)
        # Keep the primary audio-mass diagnostic aligned with the MAPO reduction path:
        # same loss-type aggregation, restricted to the POS-gated token subset.
        audio_log_penalty = -torch.log((audio_mass + _MAPO_ATTN_LOG_EPS).clamp(min=_MAPO_ATTN_LOG_EPS))
        # KL shape diagnostic: compute KL per-token between frozen nu_tilde target and actual dist.
        _cmask_f_diag = completion_mask.float()
        _nu_tilde_diag = nu_tilde.detach().float() if isinstance(nu_tilde, torch.Tensor) else _cmask_f_diag
        _target_diag = _nu_tilde_diag * _cmask_f_diag
        _target_diag = _target_diag / _target_diag.sum(-1, keepdim=True).clamp(min=1e-8)
        _actual_diag = audio_mass.float() * _cmask_f_diag
        _actual_diag = _actual_diag / _actual_diag.sum(-1, keepdim=True).clamp(min=1e-8)
        _kl_diag = (_target_diag * (
            torch.log(_target_diag.clamp(min=_MAPO_ATTN_LOG_EPS))
            - torch.log(_actual_diag.clamp(min=_MAPO_ATTN_LOG_EPS))
        ) * _cmask_f_diag).sum(-1)  # [B]
        attn_kl_mean = _kl_diag.mean()
        _total_mass_diag = (audio_mass.float() * (_target_diag > 0).float()).sum(-1)  # [B], POS-gated
        attn_mass_penalty_mean = -torch.log(_total_mass_diag.clamp(min=_MAPO_ATTN_LOG_EPS)).mean()
        failed_token_gate = pos_gate * task_failed.unsqueeze(-1)
        success_token_gate = pos_gate * (1.0 - task_failed.unsqueeze(-1))
        # Entropy filter coverage: fraction of completion tokens zeroed out by the filter.
        _efm = data.get('mapo_cached_entropy_filter_mask')
        if isinstance(_efm, torch.Tensor) and _efm.shape == completion_mask_f.shape:
            entropy_filter_frac = (
                completion_mask_f.sum() - (_efm * completion_mask_f).sum()
            ) / valid_token_denom
        else:
            entropy_filter_frac = torch.zeros((), dtype=torch.float32, device=loss.device)
        custom_metrics = {
            'completions/mean_length': total_lengths.float().mean(),
            'completions/max_length': total_lengths.float().max(),
            'completions/min_length': total_lengths.float().min(),
            'mapo/adv_abs_mean': adv_abs_mean,
            'mapo/adv_scale_mean': adv_scale_mean,
            'mapo/adv_floor_applied_frac': adv_floor_applied_frac,
            'mapo/adv_floor_lift_mean': adv_floor_lift_mean,
            'mapo/effective_task_gate_mean': effective_task_gate_mean,
            'mapo/attn_prefactor_mean': attn_prefactor_mean,
            'mapo/mask_weight_max': mask_weights.max(),
            'mapo/mask_weight_min': mask_weight_min,
            'mapo/delta_h_abs_mean': (omega_raw * completion_mask_f).sum() / valid_token_denom,
            'mapo/attn_loss': attn_loss.clone().detach(),
            'mapo/attn_kl_mean': attn_kl_mean.clone().detach(),
            'mapo/attn_mass_penalty_mean': attn_mass_penalty_mean.clone().detach(),
            'mapo/audio_mass_mean': self._reduce_token_mean(
                audio_mass, completion_mask, self.loss_type, gate_weights=pos_gate),
            'mapo/audio_mass_mean_failed_only': self._reduce_token_mean(
                audio_mass, completion_mask, self.loss_type, gate_weights=failed_token_gate),
            'mapo/audio_mass_mean_success_only': self._reduce_token_mean(
                audio_mass, completion_mask, self.loss_type, gate_weights=success_token_gate),
            'mapo/audio_mass_abs_mean': self._reduce_token_mean(
                audio_mass.abs(), completion_mask, self.loss_type, gate_weights=pos_gate),
            'mapo/audio_log_penalty_mean': self._reduce_token_mean(
                audio_log_penalty, completion_mask, self.loss_type, gate_weights=pos_gate),
            'mapo/audio_mass_max': audio_mass.abs().max(),
            'mapo/task_fail_frac': task_failed.mean(),
            'mapo/pos_gate_on_frac': (pos_gate * completion_mask_f).sum() / valid_token_denom,
            'mapo/entropy_filter_frac': entropy_filter_frac,
            'mapo/attn_only_mode': torch.tensor(
                1.0 if self.mapo_attention_only else 0.0, dtype=torch.float32, device=loss.device),
        }
        if attn_grad_probe_metrics:
            custom_metrics.update(attn_grad_probe_metrics)
        suppressed_diag_metrics = {
            'attn_hook_modules',
            'attn_tensor_updates',
            'attn_tp_world_size',
            'attn_pp_world_size',
            'attn_core_calls',
            'attn_core_success',
            'attn_core_has_tensor',
            'attn_target_modules',
            'attn_layers_used',
            'attn_available',
            'audio_mask_available',
        }
        for key, value in attn_diagnostics.items():
            metric_name = key[5:] if key.startswith('mapo_') else key
            if metric_name in suppressed_diag_metrics:
                continue
            custom_metrics[f'mapo/{metric_name}'] = torch.tensor(value, device=loss.device, dtype=torch.float32)

        if self.beta != 0.0 and per_token_kl is not None:
            kl_value = (per_token_kl * completion_mask_f).sum() / valid_token_denom
            avg_metric['kl'] = kl_value.clone().detach()

        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        completion_token_count = completion_mask_f.sum().clamp(min=1.0)
        if self.loss_type == 'cispo':
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            cispo_clip_ratio = (is_cispo_clipped.float() * completion_mask_f).sum() / completion_token_count
            self._metrics[mode]['cispo_clip_ratio'].append(cispo_clip_ratio)
        elif self.loss_type == 'sapo':
            pass
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_1_for_metrics = torch.exp(log_importance_weights)
            is_low_clipped = (coef_1_for_metrics < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1_for_metrics > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            low_clip = (is_low_clipped.float() * completion_mask_f).sum() / completion_token_count
            high_clip = (is_high_clipped.float() * completion_mask_f).sum() / completion_token_count
            is_region_clipped = is_low_clipped | is_high_clipped
            clip_ratio = (is_region_clipped.float() * completion_mask_f).sum() / completion_token_count
            weighted_clip_ratio = (mask_weights * is_region_clipped.float()).sum() / mask_weights.sum().clamp(min=1.0)

            reasoning_unclipped_mask = (mask_weights < 1.0) & completion_mask & (~is_region_clipped)
            per_token_grad_abs = (coef_1_for_metrics * advantages.unsqueeze(1)).abs()
            unclipped_grad_abs_reasoning = (
                per_token_grad_abs * reasoning_unclipped_mask.float()).sum() / reasoning_unclipped_mask.float().sum().clamp(
                    min=1.0)

            gathered_low_clip = gather(
                low_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            gathered_high_clip = gather(
                high_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            self._metrics[mode]['clip_ratio/low_mean'].append(low_clip.item())
            self._metrics[mode]['clip_ratio/high_mean'].append(high_clip.item())
            self._metrics[mode]['clip_ratio/region_mean'].append(clip_ratio.item())
            self._metrics[mode]['clip_ratio/weighted_region_mean'].append(weighted_clip_ratio.item())
            custom_metrics['clip_ratio/low_min'] = gathered_low_clip.min()
            custom_metrics['clip_ratio/high_max'] = gathered_high_clip.max()
            custom_metrics['clip_ratio/weighted_region'] = weighted_clip_ratio
            custom_metrics['mapo/unclipped_grad_abs_mean_reasoning'] = unclipped_grad_abs_reasoning

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
