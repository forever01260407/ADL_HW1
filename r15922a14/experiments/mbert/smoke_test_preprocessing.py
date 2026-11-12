from __future__ import annotations

from transformers import AutoTokenizer

from common import (
    _head_tail,
    _pack_recent_complete_turns,
    encode_record,
    resolve_preprocess_config,
    serialize_context,
)


def legacy_reference(tokenizer, context, utterance, max_length):
    """Reference implementation of the v0.1 context-first token-tail behavior."""
    utterance_ids = tokenizer.encode(
        utterance, add_special_tokens=False, verbose=False
    )
    context_text = serialize_context(context)
    context_ids = (
        tokenizer.encode(context_text, add_special_tokens=False, verbose=False)
        if context_text
        else []
    )
    single_budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    pair_budget = max_length - tokenizer.num_special_tokens_to_add(pair=True)
    if not context_ids or len(utterance_ids) >= pair_budget:
        return tokenizer.prepare_for_model(
            _head_tail(utterance_ids, single_budget),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
    context_budget = pair_budget - len(utterance_ids)
    return tokenizer.prepare_for_model(
        context_ids[-context_budget:],
        pair_ids=utterance_ids,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/xlm-roberta-base", use_fast=True
    )
    context = [
        {"role": "user", "content": "我想了解這個方案。"},
        {"role": "assistant", "content": "這是產品規格與服務內容。"},
        {"role": "user", "content": "競品比較便宜，而且這超出預算。"},
    ]
    utterance = "我需要再想一下，也想先確認最後價格。"

    legacy = encode_record(tokenizer, context, utterance, max_length=64)
    reference = legacy_reference(tokenizer, context, utterance, max_length=64)
    assert legacy["input_ids"] == reference["input_ids"]
    assert resolve_preprocess_config({}) == {
        "preprocess_version": 1,
        "input_order": "context_first",
        "context_max_tokens": None,
        "turn_aware_context": False,
        "explicit_role_markers": False,
    }

    no_context = encode_record(
        tokenizer,
        context,
        utterance,
        max_length=64,
        input_order="utterance_first",
        context_max_tokens=0,
        turn_aware_context=True,
        explicit_role_markers=True,
    )
    empty_context = encode_record(
        tokenizer,
        [],
        utterance,
        max_length=64,
        input_order="utterance_first",
        context_max_tokens=0,
        turn_aware_context=True,
        explicit_role_markers=True,
    )
    assert no_context["input_ids"] == empty_context["input_ids"]

    for budget in [0, 1, 16, 32, None]:
        encoded = encode_record(
            tokenizer,
            context,
            utterance,
            max_length=64,
            input_order="utterance_first",
            context_max_tokens=budget,
            turn_aware_context=True,
            explicit_role_markers=True,
        )
        assert len(encoded["input_ids"]) <= 64

    context_first = encode_record(
        tokenizer,
        context,
        utterance,
        max_length=64,
        input_order="context_first",
        context_max_tokens=32,
        turn_aware_context=True,
    )
    utterance_first = encode_record(
        tokenizer,
        context,
        utterance,
        max_length=64,
        input_order="utterance_first",
        context_max_tokens=32,
        turn_aware_context=True,
    )
    assert context_first["input_ids"] != utterance_first["input_ids"]
    assert "token_type_ids" not in utterance_first

    mbert_tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-multilingual-cased", use_fast=True
    )
    mbert_pair = encode_record(
        mbert_tokenizer,
        context,
        utterance,
        max_length=64,
        input_order="utterance_first",
        context_max_tokens=32,
        turn_aware_context=True,
        explicit_role_markers=True,
    )
    assert "token_type_ids" in mbert_pair
    assert len(mbert_pair["token_type_ids"]) == len(mbert_pair["input_ids"])
    assert 0 in mbert_pair["token_type_ids"]
    assert 1 in mbert_pair["token_type_ids"]

    newest_text = "user: 競品比較便宜，而且這超出預算。"
    newest_ids = tokenizer.encode(newest_text, add_special_tokens=False, verbose=False)
    packed = _pack_recent_complete_turns(
        tokenizer,
        context,
        budget=len(newest_ids),
        explicit_role_markers=False,
    )
    assert packed == newest_ids

    long_utterance = "很長的當前訊息 " * 200
    long_encoded = encode_record(
        tokenizer,
        context,
        long_utterance,
        max_length=64,
        input_order="utterance_first",
        context_max_tokens=32,
        turn_aware_context=True,
    )
    assert len(long_encoded["input_ids"]) == 64

    print("XLM-R/mBERT preprocessing compatibility and boundary tests passed.")


if __name__ == "__main__":
    main()
