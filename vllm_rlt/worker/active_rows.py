"""Shared active-row view for token-sparse prefill (RFC #156, sections 3.1, 6.1, 6.2).

One host-side answer to "which rows of this loop execute", for three consumers:

- token-sparse prefill gathers the active rows of a loop and scatters results back;
  inactive rows keep their previous hidden state and inherit the previous depth's KV;
- sparse decode attention builds its per-row plan (``SparseDecodePlan``, #157) in
  execution order with ``ActiveRows.take``; generated-token rows are never selected;
- the scheduler charges ``SelectionWindows.execution_rows`` instead of the logical span.

Windows have absolute boundaries (multiples of the window length, clipped to the
prompt), independent of the batch, the chunk and the prefix-cache hit. A sparse
depth's active set is nested in the previous depth's set and holds exactly the
window budget, so every shape follows from the budgets without a device readback.
"""

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_rlt.config import TokenSparsePrefillConfig
from vllm_rlt.request import ExecutionBatch, Request


def window_budget(window_len: int, retention: float) -> int:
    """Active rows of one window: ``max(1, ceil(f * window_len))``, as FlashLoop ``aeaadee``."""
    return min(window_len, max(1, math.ceil(window_len * retention)))


class SelectionWindows:
    """Host-known window geometry and per-depth budgets, shared by scheduler and runner."""

    def __init__(
        self, config: TokenSparsePrefillConfig, *, prefill_chunk_size: int, total_depths: int
    ):
        window = config.selection_window or prefill_chunk_size
        # A disabled group never selects, so only an enabled one must fit the stage.
        if config.enabled and prefill_chunk_size % window:
            raise ValueError("prefill chunks must contain whole selection windows")
        if config.enabled and len(config.retention_by_loop) != total_depths:
            raise ValueError("retention_by_loop needs one fraction per recurrent depth")
        self.config = config
        self.window = window

    def sparse(self, depth: int) -> bool:
        """Whether recurrent depth ``depth`` runs a subset of each window's prompt rows."""
        return self.config.enabled and 0 <= depth and self.config.retention_by_loop[depth] < 1

    def window_len(self, prompt_len: int, window_start: int) -> int:
        if window_start % self.window or not 0 <= window_start < prompt_len:
            raise ValueError("window_start must be a window boundary inside the prompt")
        return min(self.window, prompt_len - window_start)

    def budget(self, depth: int, window_len: int) -> int:
        if not self.sparse(depth):
            return window_len
        return window_budget(window_len, self.config.retention_by_loop[depth])

    def windows(self, prompt_len: int, start: int, count: int):
        """Yield ``(window_start, window_len, overlap)`` for the prompt rows of a span."""
        position, end = start, min(start + count, prompt_len)
        while position < end:
            window_start = position - position % self.window
            window_len = min(self.window, prompt_len - window_start)
            overlap_end = min(end, window_start + window_len)
            yield window_start, window_len, overlap_end - position
            position = overlap_end

    def execution_rows(self, prompt_len: int, depth: int | None, start: int, count: int) -> int:
        """Rows a span executes at ``depth``: exact for whole windows, a bound for fragments.

        A fragment's active count depends on which rows were selected, so it is
        charged ``min(overlap, budget)``. Dense depths, generated-token rows and
        ``depth=None`` (PRELUDE, or atomic PREFILL whose dense loops set the peak)
        charge every row.
        """
        if depth is None or not self.sparse(depth):
            return count
        rows = max(0, start + count - max(start, prompt_len))
        for _, window_len, overlap in self.windows(prompt_len, start, count):
            rows += min(overlap, self.budget(depth, window_len))
        return rows


def select_active(
    scores: torch.Tensor, budget: int, eligible: torch.Tensor | None = None
) -> torch.Tensor:
    """Window-local active rows of the next sparse depth, ascending (int64).

    ``scores`` holds one FP32 score per window row. Only ``eligible`` rows (the
    previous sparse depth's active set; ``None`` after a dense depth) compete, so
    active sets stay nested. Equal scores keep the smaller position, as
    ``select_keys`` does for sparse decode, so packing and batch order cannot change
    the result. No row is reserved for the window's or the prompt's final token.
    """
    if scores.ndim != 1 or scores.dtype != torch.float32:
        raise ValueError("scores must be one FP32 value per window row")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    candidates = (
        torch.arange(len(scores), device=scores.device)
        if eligible is None
        else eligible.to(device=scores.device, dtype=torch.int64)
    )
    if not 1 <= budget <= len(candidates):
        raise ValueError("budget must be between 1 and the number of eligible rows")
    order = torch.argsort(scores.index_select(0, candidates), descending=True, stable=True)
    return candidates.index_select(0, order[:budget]).sort().values


@dataclass(frozen=True)
class ActiveRows:
    """Borrowed execution order of one recurrent loop.

    ``gather[i]`` is the logical row (``ExecutionBatch.addresses()`` order) that
    executes as row ``i``; ``positions[i]`` is its absolute position for RoPE and the
    causal length. Logical rows missing from ``gather`` are inactive. ``split``
    separates generated-token rows ``[0, split)`` from active prompt rows; ``None``
    means the identity order of a loop with no inactive row.
    """

    gather: tuple[int, ...]
    positions: tuple[int, ...]
    num_logical_rows: int
    split: int | None = None

    @property
    def dense(self) -> bool:
        return self.split is None

    @property
    def num_rows(self) -> int:
        return len(self.gather)

    @property
    def inactive(self) -> tuple[int, ...]:
        """Logical rows that skip this loop, ascending: the KV inheritance input."""
        executed = set(self.gather)
        return tuple(row for row in range(self.num_logical_rows) if row not in executed)

    def take(self, values: Sequence) -> list:
        """Reorder one host value per logical row (e.g. reference modes) into execution order."""
        if len(values) != self.num_logical_rows:
            raise ValueError("take needs one value per logical row")
        return [values[row] for row in self.gather]

    def index(self, device=None) -> torch.Tensor:
        """``gather`` as an int64 index for ``index_select`` and ``index_copy_``."""
        return torch.tensor(self.gather, dtype=torch.int64, device=device)


class PrefillSelections:
    """Runner-owned nested active sets, keyed by request, absolute window and depth.

    A window's set for a sparse depth is published once the previous depth covers
    the whole window; ``view`` rejects a sparse row whose window has no set yet.
    ``release`` drops a request's sets on completion, abort or preemption, so a
    rewound request selects again from fresh scores.
    """

    def __init__(self, windows: SelectionWindows):
        self.windows = windows
        self._active: dict[tuple[str, int, int], tuple[int, ...]] = {}

    def active(self, request_id: str, window_start: int, depth: int) -> tuple[int, ...] | None:
        return self._active.get((request_id, window_start, depth))

    def publish(
        self, request: Request, window_start: int, depth: int, active: Sequence[int]
    ) -> tuple[int, ...]:
        if not self.windows.sparse(depth):
            raise ValueError(f"depth {depth} runs every prompt row")
        window_len = self.windows.window_len(len(request.prompt_token_ids), window_start)
        rows = tuple(int(row) for row in active)
        if len(rows) != self.windows.budget(depth, window_len):
            raise ValueError("an active set must hold exactly the window budget")
        if rows[0] < 0 or rows[-1] >= window_len or any(b <= a for a, b in zip(rows, rows[1:])):
            raise ValueError("active rows must be ascending window-local rows")
        if self.windows.sparse(depth - 1):
            prior = self.active(request.request_id, window_start, depth - 1)
            if prior is None or not set(rows) <= set(prior):
                raise ValueError("an active set must be nested in the previous depth's set")
        self._active[(request.request_id, window_start, depth)] = rows
        return rows

    def select(
        self, request: Request, window_start: int, depth: int, scores: torch.Tensor
    ) -> tuple[int, ...]:
        """Select and publish one window's active rows for ``depth`` from its FP32 scores."""
        window_len = self.windows.window_len(len(request.prompt_token_ids), window_start)
        if len(scores) != window_len:
            raise ValueError("scores need one value per window row")
        # A missing previous set leaves every row eligible; publish then rejects it.
        prior = self.active(request.request_id, window_start, depth - 1)
        eligible = None if prior is None else torch.tensor(prior, dtype=torch.int64)
        chosen = select_active(scores, self.windows.budget(depth, window_len), eligible)
        return self.publish(request, window_start, depth, chosen.tolist())

    def release(self, request_id: str) -> None:
        for key in [key for key in self._active if key[0] == request_id]:
            del self._active[key]

    def view(self, batch: ExecutionBatch) -> ActiveRows:
        """Execution order of one recurrent loop of ``batch``.

        With no inactive row the view is the identity, so dense loops skip gather
        and scatter. Otherwise generated-token rows come first and active prompt rows
        follow, each in logical order (section 6.2). A row is a prompt row when its
        position is inside the prompt, the same rule sparse decode uses in #157.
        """
        if batch.loop_count > 1:
            raise ValueError("bind one loop with ExecutionBatch.at_loop first")
        tokens, prompt, row = [], [], 0
        for span in batch.spans:
            request, prompt_len = span.request, len(span.request.prompt_token_ids)
            for position in range(span.token_start, span.token_start + span.token_count):
                if position >= prompt_len:
                    tokens.append((row, position))
                elif self._is_active(request.request_id, span.depth, position):
                    prompt.append((row, position))
                row += 1
        order = tokens + prompt
        split = len(tokens) if len(order) < row else None
        if split is None:
            order.sort()
        return ActiveRows(
            gather=tuple(index for index, _ in order),
            positions=tuple(position for _, position in order),
            num_logical_rows=row,
            split=split,
        )

    def _is_active(self, request_id: str, depth: int, position: int) -> bool:
        if not self.windows.sparse(depth):
            return True
        window_start = position - position % self.windows.window
        rows = self.active(request_id, window_start, depth)
        if rows is None:
            raise ValueError(f"window {window_start} of {request_id!r} has no active set yet")
        local = position - window_start
        index = bisect_left(rows, local)
        return index < len(rows) and rows[index] == local
