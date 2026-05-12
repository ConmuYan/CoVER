from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evidence.json_utils import parse_llm_err
from evidence.prompt import build_llm_messages, build_retry_messages
from evidence.schema import ERR

logger = logging.getLogger(__name__)


class OfflineLLMTeacher:
    def __init__(
        self,
        backend: str,
        model_name_or_path: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        max_new_tokens: int = 256,
        timeout: int = 60,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_verifier_retry: bool = False,
        max_verifier_retries: int = 1,
        max_parse_retries: int = 1,
    ):
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.enable_verifier_retry = enable_verifier_retry
        self.max_verifier_retries = max_verifier_retries
        self.max_parse_retries = max_parse_retries

        self._model = None
        self._tokenizer = None

        if backend == "mock":
            pass
        elif backend == "transformers_local":
            self._load_local_model()
        elif backend == "openai_compatible":
            raise NotImplementedError("openai_compatible backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _load_local_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required for transformers_local backend")

        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required for transformers_local backend")

        path = Path(self.model_name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        logger.info("Loading tokenizer from %s", path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )

        logger.info("Loading model from %s", path)
        torch_dtype = self.torch_dtype
        if torch_dtype == "auto":
            torch_dtype = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()
        logger.info("Model loaded successfully")

    def generate(self, payload: dict[str, Any]) -> tuple[ERR | None, dict[str, Any]]:
        messages = build_llm_messages(payload)
        node_id = payload.get("node_id", 0)

        metadata: dict[str, Any] = {
            "backend": self.backend,
            "model_name_or_path": self.model_name_or_path,
            "node_id": node_id,
            "retry_count": 0,
            "parsed_ok": False,
            "parse_error": None,
            "raw_output": None,
        }

        if self.backend == "mock":
            return self._generate_mock(payload, messages, metadata)
        elif self.backend == "transformers_local":
            return self._generate_local(messages, metadata)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def generate_with_verifier_retry(
        self,
        payload: dict[str, Any],
        verifier,
        card,
    ) -> tuple[ERR | None, dict[str, Any]]:
        node_id = payload.get("node_id", 0)

        all_attempts: list[dict[str, Any]] = []
        err, metadata = self.generate(payload)
        all_attempts.append(metadata.copy())

        if err is None:
            metadata["attempts"] = all_attempts
            metadata["final_status"] = "parse_failed"
            return None, metadata

        accepted, reasons = verifier.verify(err, card)
        if accepted:
            metadata["attempts"] = all_attempts
            metadata["final_status"] = "accepted"
            metadata["accepted_after_retry"] = False
            metadata["verifier_retries"] = 0
            return err, metadata

        for retry_idx in range(self.max_verifier_retries):
            retry_messages = build_retry_messages(payload, err, reasons)
            retry_metadata: dict[str, Any] = {
                "backend": self.backend,
                "model_name_or_path": self.model_name_or_path,
                "node_id": node_id,
                "retry_count": retry_idx + 1,
                "attempt_type": "verifier_retry",
                "parsed_ok": False,
                "parse_error": None,
                "raw_output": None,
            }

            if self.backend == "mock":
                err_retry, retry_metadata = self._generate_mock(payload, retry_messages, retry_metadata)
            elif self.backend == "transformers_local":
                err_retry, retry_metadata = self._generate_local(retry_messages, retry_metadata)
            else:
                break

            all_attempts.append(retry_metadata.copy())

            if err_retry is None:
                continue

            accepted_retry, reasons_retry = verifier.verify(err_retry, card)
            if accepted_retry:
                metadata["attempts"] = all_attempts
                metadata["final_status"] = "accepted_after_retry"
                metadata["accepted_after_retry"] = True
                metadata["verifier_retries"] = retry_idx + 1
                return err_retry, metadata

            err = err_retry
            reasons = reasons_retry

        metadata["attempts"] = all_attempts
        metadata["final_status"] = "rejected"
        metadata["reject_reasons"] = reasons
        metadata["verifier_retries"] = self.max_verifier_retries
        return err, metadata

    def _generate_mock(
        self, payload: dict[str, Any], messages: list[dict], metadata: dict
    ) -> tuple[ERR | None, dict]:
        reasoning = payload.get("reasoning", {})
        detector_signal = reasoning.get("detector_signal", "normal")
        detector_signal_strength = reasoning.get("detector_signal_strength", "weak")
        feature_discrepancy = reasoning.get("feature_neighbor_discrepancy", "low")
        neighbor_consistency = reasoning.get("neighbor_consistency", "high")
        degree_level = reasoning.get("degree_level", "medium")

        if detector_signal_strength == "strong" and "high_frequency_response" in detector_signal:
            risk_type = "spectral_anomaly"
            supporting = ["detector_signal", "detector_signal_strength"]
        elif feature_discrepancy == "high":
            risk_type = "feature_structure_conflict"
            supporting = ["feature_neighbor_discrepancy"]
        elif neighbor_consistency == "low":
            risk_type = "camouflage_neighbor"
            supporting = ["neighbor_consistency"]
        elif detector_signal_strength == "strong":
            risk_type = "structural_discrepancy"
            supporting = ["detector_signal", "detector_signal_strength"]
        else:
            risk_type = "weak_or_uncertain_evidence"
            supporting = ["degree_level"]

        counter = ["counter_signal"]

        mock_output = json.dumps({
            "risk_type": risk_type,
            "supporting_evidence": supporting,
            "counter_evidence": counter,
            "summary": f"Mock: {risk_type}",
        })

        metadata["raw_output"] = mock_output
        metadata["parsed_ok"] = True

        err = ERR(
            node_id=payload.get("node_id", 0),
            risk_type=risk_type,
            supporting_evidence=supporting,
            counter_evidence=counter,
            summary=f"Mock: {risk_type}",
        )

        return err, metadata

    def _generate_local(
        self, messages: list[dict], metadata: dict
    ) -> tuple[ERR | None, dict]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        for attempt in range(self.max_retries):
            metadata["retry_count"] = attempt + 1

            try:
                raw_output = self._call_model(messages)
                metadata["raw_output"] = raw_output

                err = parse_llm_err(raw_output, metadata["node_id"])
                metadata["parsed_ok"] = True
                return err, metadata

            except Exception as e:
                metadata["parse_error"] = str(e)
                logger.warning("Attempt %d failed: %s", attempt + 1, e)

        return None, metadata

    def _call_model(self, messages: list[dict]) -> str:
        import torch

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                input_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            input_text = "\n".join(m["content"] for m in messages)

        inputs = self._tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self._model.device)

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=pad_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw_output.strip()
