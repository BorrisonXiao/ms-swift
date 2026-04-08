# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_TARGET_POS_TAGS = ("NOUN", "VERB", "ADJ")
_NLTK_READY = False


def normalize_target_pos_tags(tags: Optional[Iterable[str]]) -> Set[str]:
    values: Set[str] = set()
    if tags is None:
        tags = DEFAULT_TARGET_POS_TAGS
    for tag in tags:
        normalized = str(tag or "").strip().upper()
        if normalized:
            values.add(normalized)
    if not values:
        values.update(DEFAULT_TARGET_POS_TAGS)
    return values


def _is_punctuation_piece(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return all((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)


def _normalize_word_for_tagging(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    while value and _is_punctuation_piece(value[0]):
        value = value[1:]
    while value and _is_punctuation_piece(value[-1]):
        value = value[:-1]
    return value.strip()


def _map_penn_to_coarse(tag: str) -> str:
    value = str(tag or "").upper()
    if value in {".", ",", ":", "``", "''", "-LRB-", "-RRB-", "#", "$"}:
        return "PUNCT"
    if value.startswith("NN"):
        return "NOUN"
    if value.startswith("VB"):
        return "VERB"
    if value.startswith("JJ"):
        return "ADJ"
    if value.startswith("RB"):
        return "ADV"
    if value in {"PRP", "PRP$", "WP", "WP$"}:
        return "PRON"
    if value in {"DT", "PDT", "WDT"}:
        return "DET"
    if value in {"IN", "TO"}:
        return "ADP"
    if value == "CC":
        return "CONJ"
    if value == "CD":
        return "NUM"
    if value == "RP":
        return "PART"
    return "X"


def _merge_token_units(token_texts: Sequence[str]) -> List[dict]:
    units: List[dict] = []
    current_text = ""
    current_indices: List[int] = []

    def _flush_current():
        nonlocal current_text
        if not current_indices:
            current_text = ""
            return
        units.append(
            {
                "text": current_text,
                "token_indices": list(current_indices),
                "kind": "word",
            }
        )
        current_text = ""
        current_indices.clear()

    for idx, token_text in enumerate(token_texts):
        raw = str(token_text or "")
        if raw == "":
            continue
        stripped = raw.strip()
        if stripped == "":
            _flush_current()
            continue
        if raw[0].isspace():
            _flush_current()

        if _is_punctuation_piece(stripped):
            _flush_current()
            units.append({"text": stripped, "token_indices": [idx], "kind": "punct"})
            continue

        current_text += stripped
        current_indices.append(idx)

    _flush_current()
    return units


def _decode_token_texts(token_ids: Sequence[int], tokenizer) -> List[str]:
    token_ids = [int(v) for v in token_ids]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            if isinstance(raw_tokens, (list, tuple)) and len(raw_tokens) == len(token_ids):
                token_texts: List[str] = []
                for token_text in raw_tokens:
                    text = str(token_text or "")
                    # SentencePiece/byte-level markers to readable spaces.
                    text = text.replace("▁", " ").replace("Ġ", " ")
                    token_texts.append(text)
                return token_texts
        except Exception:
            pass
    token_texts = []
    for token_id in token_ids:
        text = ""
        if hasattr(tokenizer, "decode"):
            try:
                text = tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                text = ""
        token_texts.append(str(text or ""))
    return token_texts


def _ensure_nltk_ready():
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        import nltk
    except Exception as exc:
        raise RuntimeError("NLTK is required for MAPO POS gating.") from exc

    def _probe_pos_tag():
        return nltk.pos_tag(["probe"])

    try:
        _probe_pos_tag()
    except LookupError:
        for resource in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                continue
        _probe_pos_tag()
    _NLTK_READY = True


def _pos_tag_words(words: Sequence[str]) -> List[Tuple[str, str]]:
    _ensure_nltk_ready()
    import nltk

    return list(nltk.pos_tag(list(words)))


def build_pos_token_gate(
    token_ids: Sequence[int],
    tokenizer,
    target_pos_tags: Optional[Iterable[str]] = None,
) -> Tuple[List[float], bool]:
    token_ids = [int(v) for v in (token_ids or [])]
    if not token_ids:
        return [], False

    target_pos = normalize_target_pos_tags(target_pos_tags)

    try:
        token_texts = _decode_token_texts(token_ids, tokenizer)
        units = _merge_token_units(token_texts)

        lexical_unit_indices: List[int] = []
        lexical_words: List[str] = []
        for idx, unit in enumerate(units):
            if unit.get("kind") == "punct":
                continue
            normalized_word = _normalize_word_for_tagging(unit.get("text", ""))
            if not normalized_word:
                continue
            lexical_unit_indices.append(idx)
            lexical_words.append(normalized_word)

        lexical_tags = _pos_tag_words(lexical_words) if lexical_words else []
        lexical_iter = iter(lexical_tags)

        gate = [0.0] * len(token_ids)
        for unit in units:
            token_indices = [int(v) for v in unit.get("token_indices", [])]
            if not token_indices:
                continue
            if unit.get("kind") == "punct":
                gate_value = 0.0
            else:
                normalized_word = _normalize_word_for_tagging(unit.get("text", ""))
                if not normalized_word:
                    gate_value = 0.0
                else:
                    try:
                        _, tag = next(lexical_iter)
                    except StopIteration:
                        tag = "X"
                    coarse = _map_penn_to_coarse(tag)
                    gate_value = 1.0 if coarse in target_pos else 0.0

            for token_idx in token_indices:
                if 0 <= token_idx < len(gate):
                    gate[token_idx] = gate_value

        # Avoid all-zero gating from tokenizer/tagger mismatch.
        if sum(gate) <= 0.0:
            return [1.0] * len(token_ids), True
        return gate, False
    except Exception:
        return [1.0] * len(token_ids), True
