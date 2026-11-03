"""Shared active-row view for token-sparse prefill (RFC #156, sections 3.1, 6.1, 6.2)."""

import pytest
import torch

from vllm_rlt.config import TokenSparsePrefillConfig
from vllm_rlt.core.scheduler import ScheduledRequest
from vllm_rlt.request import ExecutionBatch, Operation, Request, RequestPhase, StateSpan
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.active_rows import (
    PrefillSelections,
    SelectionWindows,
    select_active,
    window_budget,
)

# Four-row windows with retention [1, 1, 0.5, 0.25]: budgets 4, 4, 2, 1.
CONFIG = TokenSparsePrefillConfig(
    enabled=True, retention_by_loop=(1, 1, 0.5, 0.25), selection_window=4
)


def make_request(request_id, prompt_len, generated=0):
    request = Request(request_id, list(range(prompt_len)), SamplingParams())
    request.generated_token_ids = [0] * generated
    return request


def make_windows(config=CONFIG):
    return SelectionWindows(config, prefill_chunk_size=8, total_depths=4)


def one_loop(*spans):
    return ExecutionBatch(
        Operation.RECURRENT,
        tuple(StateSpan(request, start, count, depth) for request, start, count, depth in spans),
    )


def test_config_defaults_off_and_rejects_retention_that_is_not_nested():
    config = TokenSparsePrefillConfig()
    assert not config.enabled and config.retention_by_loop == (1.0, 1.0, 0.25, 0.10)
    for retention in ((1, 0.5, 0.25), (1, 1, 0.1, 0.25), (1, 1, 0), (1,), (1, 1, float("nan"))):
        with pytest.raises(ValueError):
            TokenSparsePrefillConfig(retention_by_loop=retention)
    with pytest.raises(ValueError):
        TokenSparsePrefillConfig(enabled=1)
    with pytest.raises(ValueError):
        TokenSparsePrefillConfig(selection_window=0)


def test_windows_are_absolute_and_budgets_follow_the_window_length():
    windows = make_windows()
    # A span starting inside a window (for example after a prefix hit) and a short last window.
    assert list(windows.windows(10, 3, 7)) == [(0, 4, 1), (4, 4, 4), (8, 2, 2)]
    assert [windows.budget(depth, 4) for depth in range(4)] == [4, 4, 2, 1]
    assert [windows.budget(depth, 2) for depth in range(4)] == [2, 2, 1, 1]
    assert window_budget(128, 0.25) == 32 and window_budget(128, 0.10) == 13
    assert window_budget(5, 0.01) == 1
    with pytest.raises(ValueError, match="whole selection windows"):
        SelectionWindows(CONFIG, prefill_chunk_size=6, total_depths=4)
    with pytest.raises(ValueError, match="one fraction per recurrent depth"):
        SelectionWindows(CONFIG, prefill_chunk_size=8, total_depths=3)
    # Disabled: any stage geometry is accepted and no depth is sparse.
    off = SelectionWindows(TokenSparsePrefillConfig(), prefill_chunk_size=128, total_depths=8)
    assert off.window == 128 and not any(off.sparse(depth) for depth in range(8))


def test_execution_rows_are_exact_for_whole_windows_and_bounded_for_fragments():
    windows = make_windows()
    assert windows.execution_rows(10, 2, 0, 8) == 4  # two whole windows, budget 2 each
    assert windows.execution_rows(10, 3, 0, 10) == 3  # 1 + 1 + 1 for the short window
    assert windows.execution_rows(10, 2, 3, 3) == 1 + 2  # min(overlap, budget) per window
    assert windows.execution_rows(10, 1, 0, 10) == 10  # dense depth
    assert windows.execution_rows(10, None, 0, 8) == 8  # PRELUDE / atomic PREFILL

    def execution_row_cost(scheduled, phase):
        # One possible Scheduler._execution_row_cost body for WCPB budget accounting.
        if phase != RequestPhase.PREFILLING_RECURRENT:
            return scheduled.token_count
        return windows.execution_rows(
            len(scheduled.request.prompt_token_ids),
            scheduled.depth,
            scheduled.token_start,
            scheduled.token_count,
        )

    chunk = ScheduledRequest(make_request("a", 10), token_start=0, token_count=8, depth=2)
    assert execution_row_cost(chunk, RequestPhase.PREFILLING_RECURRENT) == 4
    assert execution_row_cost(chunk, RequestPhase.PREFILLING_PRELUDE) == 8


def test_selection_is_nested_breaks_ties_by_position_and_forces_no_row():
    scores = torch.tensor([0.5, 0.9, 0.5, 0.1])
    assert select_active(scores, 2).tolist() == [0, 1]  # the final row is not forced in
    eligible = torch.tensor([0, 3])
    assert select_active(torch.tensor([0.0, 0.2, 0.9, 0.1]), 1, eligible).tolist() == [3]
    for bad in (scores.double(), torch.tensor([0.1, float("nan")])):
        with pytest.raises(ValueError):
            select_active(bad, 1)
    with pytest.raises(ValueError, match="eligible rows"):
        select_active(scores, 3, eligible)


def test_view_orders_generated_rows_first_and_keeps_absolute_positions():
    # Section 6.2: [P0, D0, P1, P2, D1, P3] with active P1 and P3 runs [D0, D1 | P1, P3].
    a, b, c = make_request("a", 4), make_request("b", 5, 1), make_request("c", 6, 1)
    selections = PrefillSelections(make_windows())
    selections.publish(a, 0, 2, (1, 3))
    view = selections.view(
        one_loop((a, 0, 1, 2), (b, 5, 1, 2), (a, 1, 2, 2), (c, 6, 1, 2), (a, 3, 1, 2))
    )
    assert (view.gather, view.split, view.num_rows) == ((1, 4, 2, 5), 2, 4)
    assert view.positions == (5, 6, 1, 3) and view.inactive == (0, 3)
    # Sparse decode (#157) keeps one row mode per logical row; prompt rows stay dense (0)
    # and generated rows consume (2). The view carries them into execution order.
    assert view.take([0, 2, 0, 0, 2, 0]) == [2, 2, 0, 0]
    with pytest.raises(ValueError, match="one value per logical row"):
        view.take([0, 2])


def test_dense_loops_keep_identity_and_inactive_rows_stay_bitwise_unchanged():
    a, b = make_request("a", 8), make_request("b", 4, 1)
    selections = PrefillSelections(make_windows())
    dense = selections.view(one_loop((a, 0, 8, 1), (b, 4, 1, 1)))
    assert dense.dense and dense.gather == tuple(range(9)) and dense.inactive == ()
    selections.publish(a, 0, 2, (0, 2))
    selections.publish(a, 4, 2, (1, 3))
    view = selections.view(one_loop((a, 0, 8, 2), (b, 4, 1, 2)))
    assert (view.gather, view.split, view.positions) == ((8, 0, 2, 5, 7), 1, (4, 0, 2, 5, 7))

    hidden = torch.randn(9, 3)
    before = hidden.clone()
    index = view.index()
    hidden.index_copy_(0, index, hidden.index_select(0, index) * 2 + 1)
    inactive = torch.tensor(view.inactive)
    assert torch.equal(hidden[inactive], before[inactive])
    assert torch.equal(hidden[index], before[index] * 2 + 1)


def test_select_publishes_from_scores_and_does_not_depend_on_the_batch():
    a, b = make_request("a", 6), make_request("b", 3, 1)  # a: windows [0, 4) and [4, 6)
    selections = PrefillSelections(make_windows())
    assert selections.select(a, 0, 2, torch.tensor([0.3, 0.1, 0.3, 0.2])) == (0, 2)
    assert selections.select(a, 0, 3, torch.zeros(4)) == (0,)  # ties inside the depth-2 set
    assert selections.select(a, 4, 2, torch.tensor([0.1, 0.4])) == (1,)
    assert selections.select(a, 4, 3, torch.tensor([0.0, 0.0])) == (1,)
    alone = selections.view(one_loop((a, 0, 6, 3)))
    batched = selections.view(one_loop((b, 3, 1, 3), (a, 0, 6, 3)))
    assert alone.positions == (0, 5) and batched.positions == (3, 0, 5)
    assert batched.gather == (0, 1, 6)


def test_view_and_publish_enforce_whole_window_sets_and_nesting():
    a = make_request("a", 8)
    selections = PrefillSelections(make_windows())
    with pytest.raises(ValueError, match="no active set"):
        selections.view(one_loop((a, 0, 4, 2)))
    with pytest.raises(ValueError, match="at_loop"):
        selections.view(ExecutionBatch(Operation.RECURRENT, (StateSpan(a, 0, 4, 1),), loop_count=2))
    for depth, rows, message in (
        (2, (1,), "window budget"),
        (2, (3, 1), "ascending"),
        (1, (0, 1, 2, 3), "every prompt row"),
        (3, (1,), "nested"),
    ):
        with pytest.raises(ValueError, match=message):
            selections.publish(a, 0, depth, rows)
    with pytest.raises(ValueError, match="window boundary"):
        selections.publish(a, 2, 2, (0, 1))
    selections.publish(a, 0, 2, (1, 3))
    with pytest.raises(ValueError, match="nested"):
        selections.publish(a, 0, 3, (2,))
    assert selections.publish(a, 0, 3, (3,)) == (3,)
    selections.release("a")
    assert selections.active("a", 0, 2) is None and selections.active("a", 0, 3) is None
