"""Deterministic parity across eager, compiled, and CUDA Graph decode."""

from __future__ import annotations

import unittest

import torch

from miniengine.compilation import build_paged_decode_ops
from miniengine.config import CompileConfig
from miniengine.cuda_graph import CudaGraphRunner
from miniengine.model import CausalLM, ModelConfig, PagedAttentionOps
from miniengine.paged_attention import PagedDecode


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires CUDA")
class DecodePathParityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.device = torch.device("cuda")
        self.config = ModelConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )
        self.model = CausalLM(self.config).to(self.device).eval()
        self.model.set_paged_attention_ops(
            PagedAttentionOps(
                prefill=lambda *args, **kwargs: None,
                decode=self._deterministic_attention,
            )
        )
        self.page_size = 4
        self.steps = (
            self._decode(
                input_ids=[3, 5, 7],
                cache_seqlens=[1, 2, 3],
                page_ids=[1, 2, 3],
            ),
            self._decode(input_ids=[11], cache_seqlens=[2], page_ids=[1]),
        )

    @staticmethod
    def _deterministic_attention(
        q,
        k_cache,
        v_cache,
        *,
        k,
        v,
        cache_seqlens,
        block_table,
        **_kwargs,
    ):
        page_size = k_cache.shape[1]
        logical_pages = torch.div(
            cache_seqlens, page_size, rounding_mode="floor"
        )
        offsets = torch.remainder(cache_seqlens, page_size).to(torch.long)
        page_ids = block_table.gather(
            1, logical_pages.to(torch.long).unsqueeze(1)
        ).squeeze(1)
        k_cache[page_ids, offsets] = k[:, 0]
        v_cache[page_ids, offsets] = v[:, 0]
        return q + k

    def _decode(
        self,
        input_ids: list[int],
        cache_seqlens: list[int],
        page_ids: list[int],
    ) -> PagedDecode:
        return PagedDecode(
            input_ids=torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            ).unsqueeze(1),
            position_ids=torch.tensor(
                cache_seqlens, dtype=torch.long, device=self.device
            ).unsqueeze(1),
            cache_seqlens=torch.tensor(
                cache_seqlens, dtype=torch.int32, device=self.device
            ),
            block_table=torch.tensor(
                page_ids, dtype=torch.int32, device=self.device
            ).unsqueeze(1),
        )

    def _new_kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        shape = (
            8,
            self.page_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        return [
            (
                torch.zeros(shape, device=self.device),
                torch.zeros(shape, device=self.device),
            )
        ]

    def _run_direct(self, ops, kv_caches):
        self.model.set_paged_decode_ops(ops)
        return [
            self.model.forward_paged_decode(
                step.input_ids,
                step.position_ids,
                kv_caches,
                step.cache_seqlens,
                step.block_table,
                self.config.max_position_embeddings,
            )
            for step in self.steps
        ]

    def test_logits_greedy_tokens_and_real_kv_pages_match(self) -> None:
        eager_caches = self._new_kv_caches()
        eager_logits = self._run_direct(
            build_paged_decode_ops(CompileConfig()), eager_caches
        )

        compiled_ops = build_paged_decode_ops(CompileConfig(enabled=True))
        compiled_caches = self._new_kv_caches()
        compiled_logits = self._run_direct(compiled_ops, compiled_caches)

        graph_caches = self._new_kv_caches()
        self.model.set_paged_decode_ops(compiled_ops)
        graph_runner = CudaGraphRunner(
            model=self.model,
            kv_caches=graph_caches,
            batch_sizes=(1, 4),
            page_size=self.page_size,
            max_model_len=self.config.max_position_embeddings,
            device=str(self.device),
            warmup_steps=1,
        )
        graph_logits = [graph_runner(step).clone() for step in self.steps]

        for eager, compiled, graphed in zip(
            eager_logits, compiled_logits, graph_logits
        ):
            torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(graphed, eager, rtol=1e-4, atol=1e-5)
            expected_tokens = eager[:, -1, :].argmax(dim=-1)
            torch.testing.assert_close(
                compiled[:, -1, :].argmax(dim=-1), expected_tokens
            )
            torch.testing.assert_close(
                graphed[:, -1, :].argmax(dim=-1), expected_tokens
            )

        for eager_cache, compiled_cache, graph_cache in zip(
            eager_caches, compiled_caches, graph_caches
        ):
            for eager_tensor, compiled_tensor, graph_tensor in zip(
                eager_cache, compiled_cache, graph_cache
            ):
                # Pages 1..3 belong to real requests. Graph capture and the
                # B=3-to-4 padding row may modify only scratch page zero.
                torch.testing.assert_close(compiled_tensor[1:4], eager_tensor[1:4])
                torch.testing.assert_close(graph_tensor[1:4], eager_tensor[1:4])


if __name__ == "__main__":
    unittest.main()
