"""Token sampling utilities."""

from collections.abc import Sequence

import torch

from miniengine.core import SamplingConfig


def apply_repetition_penalty(
    logits: torch.Tensor,
    output_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Penalize tokens that already appeared in the output."""
    if penalty == 1.0 or not output_ids:
        return logits
    token_ids = torch.tensor(output_ids, device=logits.device)
    scores = torch.gather(logits, -1, token_ids.unsqueeze(0))
    # Reduce probability of already-seen tokens
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits.scatter_(-1, token_ids.unsqueeze(0), scores)
    return logits


def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    output_ids: list[int] | None = None,
) -> int:
    """
    Sample one token from logits using the given SamplingConfig.

    Args:
        logits: shape (1, vocab_size), raw model output on the last position
        config: sampling configuration
        output_ids: previously generated tokens (for repetition penalty)

    Returns:
        A single token id (int).
    """
    if output_ids and config.repetition_penalty != 1.0:
        # Sampling filters below are out-of-place; clone only before the
        # in-place repetition update so unchanged logits need no copy.
        logits = logits.clone()
        logits = apply_repetition_penalty(
            logits, output_ids, config.repetition_penalty
        )
    #call .item() for single token id instead of .tolist() to avoid unnecessary list creation
    return _sample_token_ids(logits, config).item()


def apply_repetition_penalty_batch(
    logits: torch.Tensor,
    output_ids: Sequence[list[int]],
    penalty: float,
) -> torch.Tensor:
    """Apply one penalty to ragged per-row output histories.

    ``logits`` has shape ``(B, V)``.  The flattened ``(row, token)`` index
    lets one GPU advanced-indexing operation update every active history.
    """
    if penalty == 1.0:
        return logits

    row_ids: list[int] = []
    token_ids: list[int] = []
    for row, history in enumerate(output_ids):
        # A token is penalized once regardless of how often it was generated.
        #keys:elements in the history
        for token_id in dict.fromkeys(history):
            row_ids.append(row)
            token_ids.append(token_id)
    if not token_ids:
        return logits

    adjusted = logits.clone()
    rows = torch.tensor(row_ids, dtype=torch.long, device=logits.device)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    scores = adjusted[rows, tokens]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    adjusted.index_put_((rows, tokens), scores)
    return adjusted


def sample_batch(
    logits: torch.Tensor,
    config: SamplingConfig,
    output_ids: Sequence[list[int]],
) -> list[int]:
    """Choose one token per ``(B, V)`` logits row under one server policy.

    The whole batch follows the same GPU filtering path.  ``tolist()`` is the
    one required device-to-host synchronization that hands token ids to the
    CPU scheduler; do not replace it with one ``.item()`` per request.
    """
    logits = apply_repetition_penalty_batch(
        logits, output_ids, config.repetition_penalty
    )
    #.toList() to replace .item() pre request
    return _sample_token_ids(logits, config).squeeze(-1).tolist()


def _sample_token_ids(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """Return one vocabulary token id per logits row, shape ``(B, 1)``."""
    # Greedy decoding needs neither softmax nor random-number generation.
    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # Top-k narrows the vocabulary before the expensive top-p sort/softmax.
    if config.top_k > 0:
        k = min(config.top_k, logits.size(-1))
        candidate_logits, candidate_ids = torch.topk(
            logits, k, dim=-1, largest=True, sorted=True
        )
        candidate_logits = candidate_logits / config.temperature
    else:
        candidate_logits = logits / config.temperature
        candidate_ids = None

    probs: torch.Tensor | None = None
    if config.top_p < 1.0:
        if candidate_ids is None:
            candidate_logits, candidate_ids = torch.sort(
                candidate_logits, descending=True, dim=-1
            )
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        cumulative_probs = torch.cumsum(candidate_probs, dim=-1)
        # Mask tokens after the first prefix whose probability reaches top-p;
        # subtracting this token's probability always keeps at least one.
        mask = cumulative_probs - candidate_probs >= config.top_p
        # multinomial normalizes internally, so reuse the nucleus probabilities.
        probs = candidate_probs.masked_fill(mask, 0.0)

    if probs is None:
        probs = torch.softmax(candidate_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    if candidate_ids is not None:
        sampled = candidate_ids.gather(-1, sampled)
    return sampled
