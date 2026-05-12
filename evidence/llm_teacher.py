"""Offline LLM teacher with maximum performance optimizations.

Optimizations applied:
1. torch.compile for model acceleration
2. torch.cuda.amp for mixed precision inference
3. torch.inference_mode() for reduced overhead
4. Optimized batch processing with padding
5. KV-cache optimization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

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
        batch_size: int = 8,
        compile_model: bool = True,
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
        self.batch_size = batch_size
        self.compile_model = compile_model

        self._model = None
        self._tokenizer = None

        if backend == "mock":
            pass
        elif backend == "transformers_local":
            self._load_local_model()
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
            use_fast=True,
        )
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self._tokenizer.padding_side = "left"

        logger.info("Loading model from %s", path)
        torch_dtype = self.torch_dtype
        if torch_dtype == "auto":
            torch_dtype = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()
        self._model.config.use_cache = True
        
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model with torch.compile...")
                self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=True)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning("torch.compile failed: %s, using eager mode", e)
        
        logger.info("Model loaded successfully with dtype=%s", torch_dtype)

    @torch.inference_mode()
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

    @torch.inference_mode()
    def generate_batch(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[tuple[ERR | None, dict[str, Any]]]:
        if self.backend == "mock":
            return [self._generate_mock(p, build_llm_messages(p), {"backend": "mock", "node_id": p.get("node_id", 0)}) for p in payloads]
        elif self.backend == "transformers_local":
            return self._generate_batch_local(payloads)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    @torch.inference_mode()
    def _generate_batch_local(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[tuple[ERR | None, dict[str, Any]]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        results: list[tuple[ERR | None, dict[str, Any]]] = []
        total_batches = (len(payloads) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, batch_start in enumerate(range(0, len(payloads), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(payloads))
            batch_payloads = payloads[batch_start:batch_end]
            
            logger.info("Processing batch %d/%d (%d items)", batch_idx + 1, total_batches, len(batch_payloads))
            
            batch_messages = [build_llm_messages(p) for p in batch_payloads]
            batch_inputs = []
            
            for messages in batch_messages:
                if hasattr(self._tokenizer, "apply_chat_template"):
                    try:
                        input_text = self._tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                        )
                    except TypeError:
                        input_text = self._tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                else:
                    input_text = "\n".join(m["content"] for m in messages)
                batch_inputs.append(input_text)

            encoded = self._tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048, return_attention_mask=True,
            )
            
            input_ids = encoded["input_ids"].to(self._model.device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(self._model.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                outputs = self._model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0, pad_token_id=self._tokenizer.pad_token_id, use_cache=True, num_beams=1,
                )

            input_lengths = encoded["attention_mask"].sum(dim=1)
            
            for i, (payload, output, input_len) in enumerate(zip(batch_payloads, outputs, input_lengths)):
                node_id = payload.get("node_id", 0)
                metadata: dict[str, Any] = {
                    "backend": self.backend, "model_name_or_path": self.model_name_or_path,
                    "node_id": node_id, "retry_count": 0, "parsed_ok": False, "parse_error": None, "raw_output": None,
                }

                new_tokens = output[input_len:]
                raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                metadata["raw_output"] = raw_output
                
                try:
                    err = parse_llm_err(raw_output, node_id)
                    metadata["parsed_ok"] = True
                    results.append((err, metadata))
                except Exception as e:
                    metadata["parse_error"] = str(e)
                    logger.warning("Parse failed for node %d: %s", node_id, e)
                    
                    for retry in range(self.max_retries - 1):
                        metadata["retry_count"] = retry + 1
                        try:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                                single_input = input_ids[i:i+1]
                                single_mask = attention_mask[i:i+1]
                                retry_output = self._model.generate(
                                    input_ids=single_input, attention_mask=single_mask, max_new_tokens=self.max_new_tokens,
                                    temperature=self.temperature if self.temperature > 0 else None,
                                    do_sample=self.temperature > 0, pad_token_id=self._tokenizer.pad_token_id, use_cache=True,
                                )
                            new_tokens = retry_output[0][single_input.shape[1]:]
                            raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                            metadata["raw_output"] = raw_output
                            err = parse_llm_err(raw_output, node_id)
                            metadata["parsed_ok"] = True
                            break
                        except Exception as e2:
                            metadata["parse_error"] = str(e2)
                            logger.warning("Retry %d failed for node %d: %s", retry + 1, node_id, e2)
                    
                    results.append((None, metadata))

        return results

    @torch.inference_mode()
    def generate_with_verifier_retry(
        self, payload: dict[str, Any], verifier, card,
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
                "backend": self.backend, "model_name_or_path": self.model_name_or_path,
                "node_id": node_id, "retry_count": retry_idx + 1, "attempt_type": "verifier_retry",
                "parsed_ok": False, "parse_error": None, "raw_output": None,
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

    @torch.inference_mode()
    def generate_batch_with_verifier_retry(
        self, payloads: list[dict[str, Any]], verifier, cards: list,
    ) -> list[tuple[ERR | None, dict[str, Any]]]:
        if self.backend == "mock":
            results = []
            for payload, card in zip(payloads, cards):
                err, metadata = self.generate(payload)
                if err:
                    accepted, reasons = verifier.verify(err, card)
                    if accepted:
                        metadata["final_status"] = "accepted"
                    else:
                        metadata["final_status"] = "rejected"
                        metadata["reject_reasons"] = reasons
                else:
                    metadata["final_status"] = "parse_failed"
                results.append((err, metadata))
            return results
        
        batch_results = self.generate_batch(payloads)
        final_results = []
        
        retry_payloads = []
        retry_cards = []
        retry_indices = []
        
        for i, (err, metadata) in enumerate(batch_results):
            card = cards[i]
            payload = payloads[i]
            
            if err is not None:
                accepted, reasons = verifier.verify(err, card)
                if accepted:
                    metadata["attempts"] = [metadata.copy()]
                    metadata["final_status"] = "accepted"
                    metadata["accepted_after_retry"] = False
                    metadata["verifier_retries"] = 0
                    final_results.append((err, metadata))
                    continue
            
            retry_payloads.append(payload)
            retry_cards.append(card)
            retry_indices.append(i)
            final_results.append((err, metadata))
        
        for retry_idx in range(self.max_verifier_retries):
            if not retry_payloads:
                break
            
            retry_messages_list = []
            for i, payload in enumerate(retry_payloads):
                err, _ = final_results[retry_indices[i]]
                if err is None:
                    err = ERR(
                        node_id=payload.get("node_id", 0), risk_type="weak_or_uncertain_evidence",
                        supporting_evidence=[], counter_evidence=[], summary="Parse failed",
                    )
                _, reasons = verifier.verify(err, retry_cards[i])
                retry_messages_list.append(build_retry_messages(payload, err, reasons))
            
            retry_batch_results = self._generate_batch_from_messages(retry_messages_list)
            
            next_retry_payloads = []
            next_retry_cards = []
            next_retry_indices = []
            
            for j, (retry_payload, retry_card, (err_retry, retry_metadata)) in enumerate(
                zip(retry_payloads, retry_cards, retry_batch_results)
            ):
                orig_idx = retry_indices[j]
                orig_err, orig_metadata = final_results[orig_idx]
                
                if err_retry is not None:
                    accepted_retry, reasons_retry = verifier.verify(err_retry, retry_card)
                    if accepted_retry:
                        orig_metadata["attempts"] = [orig_metadata.copy(), retry_metadata.copy()]
                        orig_metadata["final_status"] = "accepted_after_retry"
                        orig_metadata["accepted_after_retry"] = True
                        orig_metadata["verifier_retries"] = retry_idx + 1
                        final_results[orig_idx] = (err_retry, orig_metadata)
                        continue
                
                next_retry_payloads.append(retry_payload)
                next_retry_cards.append(retry_card)
                next_retry_indices.append(orig_idx)
                orig_metadata["attempts"] = orig_metadata.get("attempts", [orig_metadata.copy()])
                orig_metadata["attempts"].append(retry_metadata.copy())
                final_results[orig_idx] = (err_retry if err_retry else orig_err, orig_metadata)
            
            retry_payloads = next_retry_payloads
            retry_cards = next_retry_cards
            retry_indices = next_retry_indices
        
        for i, (err, metadata) in enumerate(final_results):
            if "final_status" not in metadata:
                metadata["final_status"] = "rejected"
                metadata["reject_reasons"] = ["max_retries_exceeded"]
                metadata["verifier_retries"] = self.max_verifier_retries
        
        return final_results

    @torch.inference_mode()
    def _generate_batch_from_messages(self, messages_list: list[list[dict]]) -> list[tuple[ERR | None, dict[str, Any]]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        results: list[tuple[ERR | None, dict[str, Any]]] = []
        batch_inputs = []
        
        for messages in messages_list:
            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    input_text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                except TypeError:
                    input_text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            else:
                input_text = "\n".join(m["content"] for m in messages)
            batch_inputs.append(input_text)

        encoded = self._tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = encoded["input_ids"].to(self._model.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(self._model.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = self._model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0, pad_token_id=self._tokenizer.pad_token_id, use_cache=True,
            )

        input_lengths = encoded["attention_mask"].sum(dim=1)
        
        for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
            node_id = 0
            metadata: dict[str, Any] = {
                "backend": self.backend, "model_name_or_path": self.model_name_or_path,
                "node_id": node_id, "retry_count": 0, "parsed_ok": False, "parse_error": None, "raw_output": None,
            }

            new_tokens = output[input_len:]
            raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            metadata["raw_output"] = raw_output
            
            try:
                err = parse_llm_err(raw_output, node_id)
                metadata["parsed_ok"] = True
                results.append((err, metadata))
            except Exception as e:
                metadata["parse_error"] = str(e)
                results.append((None, metadata))

        return results

    def _generate_mock(self, payload: dict[str, Any], messages: list[dict], metadata: dict) -> tuple[ERR | None, dict]:
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
            "risk_type": risk_type, "supporting_evidence": supporting,
            "counter_evidence": counter, "summary": f"Mock: {risk_type}",
        })

        metadata["raw_output"] = mock_output
        metadata["parsed_ok"] = True

        err = ERR(
            node_id=payload.get("node_id", 0), risk_type=risk_type,
            supporting_evidence=supporting, counter_evidence=counter, summary=f"Mock: {risk_type}",
        )
        return err, metadata

    @torch.inference_mode()
    def _generate_local(self, messages: list[dict], metadata: dict) -> tuple[ERR | None, dict]:
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

    @torch.inference_mode()
    def _call_model(self, messages: list[dict]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            input_text = "\n".join(m["content"] for m in messages)

        inputs = self._tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device, non_blocking=True)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self._model.device, non_blocking=True)

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = self._model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0, pad_token_id=pad_token_id, use_cache=True,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw_output.strip()
