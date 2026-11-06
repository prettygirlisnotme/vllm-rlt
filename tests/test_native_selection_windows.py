"""Native Scheduler selection-window clipping and sparse readiness gate."""

from __future__ import annotations

import pytest

from vllm_rlt.config import SchedulerConfig, TokenSparsePrefillConfig
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.request import Request, RequestPhase
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.active_rows import SelectionWindows

TOTAL_DEPTHS = 4
CHUNK = 4
RETENTION = (1.0, 1.0, 0.75, 0.5)


class _Cache:
    max_loops = TOTAL_DEPTHS

    def ensure_capacity(self, request_id, frontier):
        return True


def _scheduler(budget=CHUNK, *, window=None, enabled=True):
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=budget,
            mode="mixed",
            prefill_chunk_size=CHUNK,
            wavefront_chunked_prefill=True,
            max_prefill_batches_before_decode=1,
        ),
        _Cache(),
        total_ut_steps=TOTAL_DEPTHS,
    )
    if window is not None:
        scheduler.token_sparse = SelectionWindows(
            TokenSparsePrefillConfig(
                enabled=enabled,
                retention_by_loop=RETENTION,
                selection_window=window,
            ),
            prefill_chunk_size=CHUNK,
            total_depths=TOTAL_DEPTHS,
        )
    return scheduler


def _request(prompt_len, *, ranges, origin=0):
    request = Request(
        "long",
        list(range(1, prompt_len + 1)),
        SamplingParams(max_tokens=1, min_loops=1, max_loops=1, ignore_eos=True),
    )
    request.phase = RequestPhase.PREFILLING_RECURRENT
    request.prefill_depth_ranges = [list(span) for span in ranges]
    request.prefill_window_origin = origin
    request.num_preluded_tokens = prompt_len
    request.num_prefilled_tokens = 0
    return request


def test_spans_stay_inside_absolute_selection_window():
    prompt_len = 8
    window = 2
    scheduler = _scheduler(window=window)
    for start in (1, 2, 3, 5, 6, 7):
        ranges = [[start, prompt_len], [0, start], [0, 0], [0, 0]]
        request = _request(prompt_len, ranges=ranges)
        task = scheduler._make_wavefront_task(
            request, RequestPhase.PREFILLING_RECURRENT, CHUNK, start=start, depth=0
        )
        selection_start = start - start % window
        selection_end = min(prompt_len, selection_start + window)
        chunk_start = start // CHUNK * CHUNK
        expected_hidden = (chunk_start // CHUNK) % TOTAL_DEPTHS * CHUNK + start - chunk_start
        assert task is not None
        assert (task.token_start, task.token_start + task.token_count) == (
            start,
            selection_end,
        )
        assert task.chunk_start == chunk_start
        assert task.prefill_hidden_row_offset == expected_hidden
        assert task.token_start + task.token_count <= selection_end


@pytest.mark.parametrize(
    ("prompt_len", "start", "previous_left", "admitted"),
    [(8, 0, 2, False), (8, 0, 4, True), (6, 4, 5, False), (6, 4, 6, True)],
)
def test_sparse_depth_waits_for_whole_previous_window(prompt_len, start, previous_left, admitted):
    scheduler = _scheduler(window=CHUNK)
    ranges = [
        [prompt_len, prompt_len],
        [previous_left, prompt_len],
        [start, previous_left],
        [0, start],
    ]
    request = _request(prompt_len, ranges=ranges)
    task = scheduler._make_wavefront_task(
        request, RequestPhase.PREFILLING_RECURRENT, CHUNK, start=start, depth=2
    )
    if admitted:
        assert task is not None
        assert task.token_count > 0
        assert task.token_start + task.token_count <= start + CHUNK
    else:
        assert task is None


@pytest.mark.parametrize("depth", (0, 1))
def test_dense_depths_may_submit_bounded_fragments(depth):
    prompt_len = 8
    scheduler = _scheduler(budget=3, window=CHUNK)
    ranges = (
        [[0, prompt_len], [0, 0], [0, 0], [0, 0]]
        if depth == 0
        else [[3, prompt_len], [0, 3], [0, 0], [0, 0]]
    )
    request = _request(prompt_len, ranges=ranges)
    task = scheduler._make_wavefront_task(
        request, RequestPhase.PREFILLING_RECURRENT, 3, start=0, depth=depth
    )
    assert task is not None
    assert task.token_count == 3


@pytest.mark.parametrize("window", (None, 2))
def test_missing_or_disabled_sparse_keeps_original_span(window):
    prompt_len = 8
    scheduler = _scheduler() if window is None else _scheduler(window=window, enabled=False)
    ranges = [[0, prompt_len], [0, 0], [0, 0], [0, 0]]
    request = _request(prompt_len, ranges=ranges)
    dense_task = scheduler._make_wavefront_task(
        request, RequestPhase.PREFILLING_RECURRENT, CHUNK, start=0, depth=0
    )
    assert dense_task is not None
    assert (dense_task.token_start, dense_task.token_count) == (0, CHUNK)
    sparse_request = _request(
        prompt_len, ranges=[[prompt_len, prompt_len], [3, prompt_len], [0, 3], [0, 0]]
    )
    sparse_task = scheduler._make_wavefront_task(
        sparse_request, RequestPhase.PREFILLING_RECURRENT, CHUNK, start=0, depth=2
    )
    assert sparse_task is not None
    assert (sparse_task.token_start, sparse_task.token_count) == (0, 3)
