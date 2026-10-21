"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights, require_local_model_dir
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Bare-bone model wrapper for single-request prefill and decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype
        local_model_path = require_local_model_dir(model_path)

        # ── Tokenizer files are read from the same local checkpoint. ────
        logger.info("Loading tokenizer from %s …", local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_path), trust_remote_code=True, local_files_only=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", local_model_path)
        config = ModelConfig.from_pretrained(str(local_model_path))
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        self.model = CausalLM(config)
        load_weights(self.model, str(local_model_path), dtype=dtype, device=device)
        self.model.eval()

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def validate_decode_admission(self, request: Request) -> None:
        """Validate the compact-KV invariant once before batched decode.

        Scheduler admission calls this after prefill has created the request's
        KV cache and appended its first output token. Keeping these checks at
        that state transition avoids repeating Python validation on each
        subsequent decode step.
        """
        if not request.output_ids:
            raise ValueError("batched decode requires a previously generated token")

        expected_layers = len(self.model.model.layers)
        if request.kv_cache is None or len(request.kv_cache) != expected_layers:
            raise ValueError("request must own one initialized KV cache per model layer")

        model_parameter = next(self.model.parameters())
        model_device = model_parameter.device
        model_dtype = model_parameter.dtype
        first_key, first_value = request.kv_cache[0]
        if first_key.ndim != 4 or first_key.shape != first_value.shape:
            raise ValueError("each request must have matching rank-four K/V tensors")

        cache_len = first_key.shape[2]
        for key, value in request.kv_cache:
            if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
                raise ValueError(
                    "each request KV cache must be matching batch-size-one K/V"
                )
            if key.shape[2] != cache_len:
                raise ValueError("all KV layers for a request must share one cache length")
            if key.device != model_device or key.dtype != model_dtype:
                raise ValueError("KV cache device and dtype must match the model")

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=request.kv_cache)
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )


    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """Run one decode token for every request in a static M1 batch.

        Each request owns a compact batch-size-one KV cache. This method
        temporarily right-pads those caches to the longest request, runs one
        model forward, then reconstructs compact independent caches. Padding
        is therefore never retained by an individual request.
        """
        if not requests:
            return []

        model_parameter = next(self.model.parameters())
        model_device = model_parameter.device
        #for Qwen3-0.6B 28 layers
        num_layers = len(requests[0].kv_cache or [])
        # Admission validates that every request has the expected layer count.
        # Original hot-path check retained for learning/debug reference:
        # if num_layers == 0:
        #     raise ValueError("batched_decode requires initialized per-layer KV caches")

        cache_lengths: list[int] = []
        for request in requests:
            # Admission establishes these compact-KV invariants once. Keep
            # the original per-step checks commented for future inspection.
            # if not request.output_ids:
            #     raise ValueError("batched_decode requires a previously generated token")
            # if request.kv_cache is None or len(request.kv_cache) != num_layers:
            #     raise ValueError("all requests must have the same initialized KV layers")
            first_key, first_value = request.kv_cache[0]
            #k、V:(1,Hkv,cache_len,head_dim)
            #for qwen3-0.6B:
            # Hkv      = 8
            # head_dim = 128
            # if first_key.ndim != 4 or first_key.shape != first_value.shape:
            #     raise ValueError("each KV cache must be matching rank-four K/V tensors")
            cache_len = first_key.shape[2]
            # for key, value in request.kv_cache:
            #     all KV layers have the same cache length and ownership.
            #key.shape[0] != 1:(1, Hkv, L_i, D)√(B, Hkv, Lmax, D)×
            #     if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
            #         raise ValueError(
            #             "each request KV cache must be matching batch-size-one K/V"
            #         )
            #     if key.shape[2] != cache_len:
            #         raise ValueError(
            #             "all KV layers for a request must share one cache length"
            #         )
            #     if key.device != model_device or key.dtype != model_dtype:
            #         raise ValueError("KV cache device and dtype must match the model")
            cache_lengths.append(cache_len)

        batch_size = len(requests)
        max_cache_len = max(cache_lengths)
        input_ids = torch.tensor(
            #id of first generated token
            [[request.output_ids[-1]] for request in requests],
            dtype=torch.long,
            device=model_device,
        )
        position_ids = torch.tensor(
            #position ids is used in RoPE position,
            #so cache lengths ,rather than padded ones length is passed
            cache_lengths, dtype=torch.long, device=model_device
        ).unsqueeze(1)

        # Equal-length KV has no padding, so every cache entry is visible.
        # Passing None also keeps the no-mask SDPA path. Mixed lengths retain
        # the original mask, whose final column is the newly decoded token.
        attention_mask: torch.Tensor | None = None
        if len(set(cache_lengths)) != 1:
            old_positions = torch.arange(max_cache_len, device=model_device).unsqueeze(0)
            #[
            #     [T, T, T, T, T],
            #     [T, T, T, F, F],
            # ]
            old_visible = old_positions < position_ids
            new_token_visible = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=model_device
            )
            attention_mask = torch.cat([old_visible, new_token_visible], dim=1)
            #(B, Lmax + 1) -> (B, 1, 1, Lmax + 1)
            attention_mask = attention_mask[:, None, None, :]

        batched_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for request, cache_len in zip(requests, cache_lengths):
                key, value = request.kv_cache[layer_idx]
                right_padding = max_cache_len - cache_len
                if right_padding:
                    #pad in sequence dim right side
                    key = F.pad(key, (0, 0, 0, right_padding))
                    value = F.pad(value, (0, 0, 0, right_padding))
                keys.append(key)
                values.append(value)
            batched_kv_caches.append(
                #cat in batch dim for all requests
                (torch.cat(keys, dim=0), torch.cat(values, dim=0))
            )

        #returned kv cache is still padded
        logits, returned_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=batched_kv_caches,
            attention_mask=attention_mask,
        )

        for request_idx, (request, cache_len) in enumerate(
            #[
            #     (0, (req_A, 3)),
            #     (1, (req_B, 5)),
            #     (2, (req_C, 2))
            # ]
            zip(requests, cache_lengths)
        ):
            compact_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for returned_key, returned_value in returned_kv_caches:
                #[batch_size, num_kv_heads, seq_len, head_dim]
                key = torch.cat(
                    [
                        #request_idx:request_idx + 1:reserve batch dim
                        returned_key[request_idx:request_idx + 1, :, :cache_len, :],
                        #-1：last token，new token,compact kv
                        returned_key[request_idx:request_idx + 1, :, -1:, :],
                    ],
                    #cat by sequence dim
                    dim=2,
                    #ensure coalesced memory layout
                ).contiguous()
                value = torch.cat(
                    [
                        returned_value[request_idx:request_idx + 1, :, :cache_len, :],
                        returned_value[request_idx:request_idx + 1, :, -1:, :],
                    ],
                    dim=2,
                ).contiguous()
                compact_kv.append((key, value))
            request.kv_cache = compact_kv

        return [
            sample_token(
                #reserve batch dim,ith request
                logits[i:i + 1, -1, :],
                request.sampling_params,
                request.output_ids,
            )
            for i, request in enumerate(requests)
        ]

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
