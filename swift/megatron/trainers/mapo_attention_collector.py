# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from swift.rlhf_trainers.utils import pad_logps_back_to_batch
from swift.utils import get_logger

logger = get_logger()

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def parse_attention_layer_spec(layer_spec: str) -> Optional[Set[int]]:
    spec = str(layer_spec or "all").strip().lower()
    if spec in {"all", "*"}:
        return None
    selected: Set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            selected.add(int(token))
        except Exception as exc:
            raise ValueError(f"Invalid layer id in mapo_attention_layers: {part!r}") from exc
    return selected


def detect_audio_token_ids(model, tokenizer=None) -> List[int]:
    token_ids = set()

    def _add_value(value):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _add_value(item)
            return
        try:
            token_ids.add(int(value))
        except Exception:
            return

    config = getattr(model, "config", None)
    thinker_config = getattr(config, "thinker_config", None) if config is not None else None
    for source in (thinker_config, config):
        if source is None:
            continue
        _add_value(getattr(source, "audio_token_index", None))
        _add_value(getattr(source, "audio_token_id", None))
        _add_value(getattr(source, "audio_token_ids", None))

    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|audio_pad|>", "<|audio_start|>", "<|audio_end|>", "<|AUDIO|>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if token_id in (None, ""):
                continue
            try:
                token_id = int(token_id)
            except Exception:
                continue
            if unk_id is not None and token_id == int(unk_id):
                continue
            token_ids.add(token_id)
    return sorted(token_ids)


def build_audio_token_mask(input_ids: Optional[torch.Tensor], audio_token_ids: Sequence[int]) -> Optional[torch.Tensor]:
    if input_ids is None:
        return None
    if not isinstance(input_ids, torch.Tensor):
        return None
    if input_ids.ndim != 2:
        return None
    if not audio_token_ids:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in audio_token_ids:
        mask = mask | (input_ids == int(token_id))
    return mask


class MAPOAttentionCollector:

    def __init__(self, attention_layers: Optional[Iterable[int]] = None,
                 head_reduce: str = 'max', layer_reduce: str = 'mean'):
        self._selected_layers: Optional[Set[int]] = None if attention_layers is None else {int(v) for v in attention_layers}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._layer_attentions: Dict[int, torch.Tensor] = {}
        self._num_hook_modules = 0
        self._num_tensor_updates = 0
        self._head_reduce = str(head_reduce).strip().lower()
        self._layer_reduce = str(layer_reduce).strip().lower()

    @staticmethod
    def _parse_layer_index(module_name: str) -> Optional[int]:
        match = _LAYER_INDEX_PATTERN.search(str(module_name))
        if match is None:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _module_is_attention(module_name: str) -> bool:
        name = str(module_name).lower()
        # Strictly target the attention module node itself, not descendants like
        # `.self_attn.q_proj`, otherwise non-attention 4D tensors can be captured.
        return (
            name.endswith(".self_attention")
            or name.endswith(".self_attn")
            or name == "self_attention"
            or name == "self_attn"
        )

    def _extract_attention_tensor(self, output) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor) and output.ndim >= 4:
            return output
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                tensor = self._extract_attention_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(output, dict):
            for key in ("attention_probs", "attn_probs", "attentions", "attention_weights", "attn_weights"):
                if key in output:
                    tensor = self._extract_attention_tensor(output[key])
                    if tensor is not None:
                        return tensor
            return None
        for attr in ("attention_probs", "attn_probs", "attentions", "attention_weights", "attn_weights"):
            value = getattr(output, attr, None)
            if value is not None:
                tensor = self._extract_attention_tensor(value)
                if tensor is not None:
                    return tensor
        return None

    def _hook(self, layer_idx: int):

        def _forward_hook(module, _inputs, output):
            tensor = self._extract_attention_tensor(output)
            if tensor is None:
                tensor = self._extract_attention_tensor(getattr(module, "_mapo_last_attention_probs", None))
            if tensor is None:
                return
            self._layer_attentions[int(layer_idx)] = tensor
            self._num_tensor_updates += 1

        return _forward_hook

    def attach(self, model):
        self.detach()
        for module_name, module in model.named_modules():
            if not self._module_is_attention(module_name):
                continue
            layer_idx = self._parse_layer_index(module_name)
            if layer_idx is None:
                continue
            if self._selected_layers is not None and layer_idx not in self._selected_layers:
                continue
            handle = module.register_forward_hook(self._hook(layer_idx))
            self._handles.append(handle)
            self._num_hook_modules += 1

    def detach(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._layer_attentions.clear()

    def reset(self):
        self._layer_attentions.clear()
        self._num_tensor_updates = 0

    @contextmanager
    def capture(self, model):
        self._layer_attentions.clear()
        self._num_hook_modules = 0
        self._num_tensor_updates = 0
        self.attach(model)
        try:
            yield self
        finally:
            self.detach()

    @staticmethod
    def _normalize_attention_tensor(attn: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(attn, torch.Tensor):
            return None
        if attn.ndim == 5:
            # [x, b, h, q, k] -> [b, h, q, k]
            attn = attn[0]
        if attn.ndim == 4:
            return attn
        if attn.ndim == 3:
            # [h, q, k] -> [1, h, q, k]
            return attn.unsqueeze(0)
        return None

    def compute_audio_attention_mass(
        self,
        audio_token_mask: Optional[torch.Tensor],
        completion_mask: Optional[torch.Tensor],
        seq_lengths: Optional[torch.Tensor] = None,
        tp_world_size: int = 1,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        tp_size = max(int(tp_world_size), 1)
        diagnostics = {
            "mapo_attn_hook_modules": float(self._num_hook_modules),
            "mapo_attn_tensor_updates": float(self._num_tensor_updates),
            "mapo_attn_layers_used": 0.0,
            "mapo_attn_available": 0.0,
            "mapo_attn_tp_world_size": float(tp_size),
        }
        if audio_token_mask is None or completion_mask is None:
            return None, diagnostics
        if not isinstance(audio_token_mask, torch.Tensor) or not isinstance(completion_mask, torch.Tensor):
            return None, diagnostics
        if audio_token_mask.ndim != 2 or completion_mask.ndim != 2:
            return None, diagnostics
        if completion_mask.numel() == 0:
            return None, diagnostics

        batch_size, seq_len = int(completion_mask.shape[0]), int(completion_mask.shape[1])
        layer_masses: List[torch.Tensor] = []
        audio_token_mask = audio_token_mask.to(device=completion_mask.device, dtype=torch.bool)
        for _, raw_attn in sorted(self._layer_attentions.items()):
            attn = self._normalize_attention_tensor(raw_attn)
            if attn is None:
                continue
            attn_batch = int(attn.shape[0])
            attn_q = int(attn.shape[2])
            attn_k = int(attn.shape[3])
            if attn_batch <= 0 or attn_q <= 0 or attn_k <= 0:
                continue

            padding_free_mode = (attn_batch == 1 and batch_size > 1 and seq_lengths is not None)
            if padding_free_mode:
                b = 1
                q = attn_q
                k = min(attn_k, int(audio_token_mask.shape[1]))
            else:
                b = min(attn_batch, batch_size)
                q = min(attn_q, seq_len)
                k = min(attn_k, seq_len, int(audio_token_mask.shape[1]))
            if b <= 0 or q <= 0 or k <= 0:
                continue
            attn = attn[:b, :, :q, :k]
            key_mask = audio_token_mask[:b, :k]
            if key_mask.shape[0] != b:
                continue
            key_mask = key_mask.unsqueeze(1).unsqueeze(2).to(dtype=attn.dtype, device=attn.device)
            # [b, h, q, k] -> per-head audio mass [b, h, q]
            per_head_mass = (attn * key_mask).sum(dim=-1)
            # Inter-head aggregation -> [b, q]
            if self._head_reduce == 'max':
                layer_mass = per_head_mass.max(dim=1).values
            else:
                layer_mass = per_head_mass.mean(dim=1) / float(tp_size)

            if layer_mass.shape[0] == batch_size and layer_mass.shape[1] == seq_len:
                layer_masses.append(layer_mass.to(device=completion_mask.device, dtype=torch.float32))
                continue

            # Padding-free mode: [1, total_tokens] -> [batch, max_seq_len].
            if layer_mass.shape[0] == 1 and seq_lengths is not None:
                try:
                    layer_mass_padded, _ = pad_logps_back_to_batch(
                        logps_rmpad=layer_mass,
                        logits_to_keep=seq_len,
                        batch_size=batch_size,
                        seq_lengths=seq_lengths,
                        pad_value=0.0)
                    layer_masses.append(layer_mass_padded.to(device=completion_mask.device, dtype=torch.float32))
                    continue
                except Exception:
                    pass

            layer_mass_padded = torch.zeros((batch_size, seq_len), dtype=layer_mass.dtype, device=layer_mass.device)
            b_pad = min(batch_size, int(layer_mass.shape[0]))
            q_pad = min(seq_len, int(layer_mass.shape[1]))
            layer_mass_padded[:b_pad, :q_pad] = layer_mass[:b_pad, :q_pad]
            layer_masses.append(layer_mass_padded.to(device=completion_mask.device, dtype=torch.float32))

        if not layer_masses:
            return None, diagnostics

        diagnostics["mapo_attn_layers_used"] = float(len(layer_masses))
        diagnostics["mapo_attn_available"] = 1.0
        stacked = torch.stack(layer_masses, dim=0)
        # Inter-layer aggregation
        if self._layer_reduce == 'max':
            return stacked.max(dim=0).values, diagnostics
        return stacked.mean(dim=0), diagnostics
