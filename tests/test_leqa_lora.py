"""Tests for LEQAModel — shape, interface, and JSON parsing.

Uses mock LoRA (no actual Qwen load) to verify the JSON output -> tensor
conversion pipeline.
"""

from __future__ import annotations

import json

import pytest
import torch

from models.leqa_lora import (
    LEQAModel,
    build_leqa_token_names,
    get_default_token_names,
    parse_leqa_json,
    parse_leqa_batch,
    format_leqa_prompt,
    build_training_example,
    audit_leqa_output,
    LEQA_CARD_FIELDS,
    LEQA_REASON_LABELS,
    LEQA_REASON_TO_ID,
)


class TestBuildTokenNames:
    def test_yelpchi_token_count(self):
        names = build_leqa_token_names(["RUR", "RSR", "RTR"])
        assert len(names) == 37  # 10 card + 3*9 relation stats

    def test_amazon_token_count(self):
        names = build_leqa_token_names(["UPU", "USU", "UVU"])
        assert len(names) == 37

    def test_default_matches_yelpchi(self):
        default = get_default_token_names()
        yelpchi = build_leqa_token_names(["RUR", "RSR", "RTR"])
        assert default == yelpchi

    def test_card_fields_first(self):
        names = get_default_token_names()
        for i, field in enumerate(LEQA_CARD_FIELDS):
            assert names[i] == field


class TestParseLeqaJson:
    def test_valid_json(self):
        token_names = get_default_token_names()
        T = len(token_names)
        entries = [
            {"token": token_names[0], "q": 0.5, "reason": "noisy"},
            {"token": token_names[1], "q": 0.8, "reason": "none"},
        ]
        raw = json.dumps({"token_quality": entries})
        q_list, reason_list = parse_leqa_json(raw, token_names)
        assert len(q_list) == T
        assert len(reason_list) == T
        assert q_list[0] == pytest.approx(0.5)
        assert q_list[1] == pytest.approx(0.8)
        assert reason_list[0] == LEQA_REASON_TO_ID["noisy"]
        assert reason_list[1] == LEQA_REASON_TO_ID["none"]
        # Unmentioned tokens get default q=1.0
        assert q_list[2] == pytest.approx(1.0)

    def test_invalid_json_returns_defaults(self):
        token_names = get_default_token_names()
        q_list, reason_list = parse_leqa_json("not json", token_names)
        assert all(q == 1.0 for q in q_list)
        assert all(r == 0 for r in reason_list)

    def test_q_clamped_to_01(self):
        token_names = get_default_token_names()
        entries = [
            {"token": token_names[0], "q": -0.5, "reason": "none"},
            {"token": token_names[1], "q": 1.5, "reason": "none"},
        ]
        raw = json.dumps({"token_quality": entries})
        q_list, _ = parse_leqa_json(raw, token_names)
        assert q_list[0] == pytest.approx(0.0)
        assert q_list[1] == pytest.approx(1.0)


class TestParseBatch:
    def test_batch_shape(self):
        token_names = get_default_token_names()
        T = len(token_names)
        texts = [
            json.dumps({"token_quality": [
                {"token": token_names[0], "q": 0.3, "reason": "contradictory"},
            ]}),
            json.dumps({"token_quality": []}),
        ]
        q_tensor, reason_tensor = parse_leqa_batch(texts, token_names)
        assert q_tensor.shape == (2, T)
        assert reason_tensor.shape == (2, T)
        assert q_tensor.dtype == torch.float32
        assert reason_tensor.dtype == torch.long


class TestLEQAModel:
    def test_num_tokens(self):
        model = LEQAModel()
        assert model.num_tokens == 37

    def test_uniform_quality(self):
        model = LEQAModel()
        q = model.uniform_quality(5)
        assert q.shape == (5, 37)
        assert (q == 1.0).all()

    def test_random_quality(self):
        model = LEQAModel()
        q = model.random_quality(5)
        assert q.shape == (5, 37)
        assert q.min() >= 0.0
        assert q.max() <= 1.0

    def test_parse_outputs(self):
        model = LEQAModel()
        raw = [json.dumps({"token_quality": [
            {"token": model.token_names[0], "q": 0.1, "reason": "insufficient"},
        ]})]
        q, r = model.parse_outputs(raw)
        assert q.shape == (1, 37)
        assert r.shape == (1, 37)
        assert q[0, 0].item() == pytest.approx(0.1)
        assert r[0, 0].item() == LEQA_REASON_TO_ID["insufficient"]


class TestAuditOutput:
    def test_clean_output_passes(self):
        assert audit_leqa_output('{"token_quality": []}')

    def test_forbidden_field_fails(self):
        assert not audit_leqa_output('base_score is 0.5')
        assert not audit_leqa_output('the confidence is high')
        assert not audit_leqa_output('ground_truth = 1')


class TestFormatting:
    def test_prompt_format(self):
        prompt = format_leqa_prompt('{"node_id": 1}')
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "evidence quality auditor" in prompt.lower()

    def test_training_example(self):
        token_names = get_default_token_names()
        q_labels = [1.0] * len(token_names)
        reason_labels = ["none"] * len(token_names)
        text = build_training_example(
            '{"node_id": 1}', token_names, q_labels, reason_labels,
        )
        assert "<|im_start|>assistant" in text
        assert "token_quality" in text
