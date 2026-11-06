"""Native shared ActiveRows pack/writeback for single-loop recurrent batches."""

from __future__ import annotations

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import ExecutionBatch, Operation, Request, StateSpan
from vllm_rlt.worker.active_rows import ActiveRows
from vllm_rlt.worker.model_runner import ModelRunner

CHUNK = 4


def _make(config, *, max_seqs=2, budget=8):
    cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
        layout="last_exited",
    )
    model = OuroForCausalLM(config).eval()
    runner = ModelRunner(
        model,
        cache,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=budget,
            mode="mixed",
            prefill_chunk_size=CHUNK,
            wavefront_chunked_prefill=True,
        ),
    )
    return model, cache, runner


def _inherit(cache, request_id, position, depth):
    for layer in range(cache.num_layers):
        keys, values = cache.read(layer, request_id, depth=depth - 1, length=position + 1)
        cache.write(
            layer,
            [request_id],
            [depth],
            [position],
            keys[position : position + 1],
            values[position : position + 1],
        )


def _seed_prompt(model, cache, request, *, inherit=()):
    ids = request.prompt_token_ids
    assert cache.allocate(request.request_id, len(ids) + CHUNK)
    hidden, _ = model.recurrent(
        model.prelude(torch.tensor(ids)),
        [request.request_id] * len(ids),
        [0] * len(ids),
        list(range(len(ids))),
        cache,
        compute_gate=False,
    )
    for position in inherit:
        _inherit(cache, request.request_id, position, 1)
    return hidden


def _seed_generated(model, cache, request, token):
    ids = request.prompt_token_ids
    assert cache.allocate(request.request_id, len(ids) + CHUNK)
    hidden, _ = model.recurrent(
        model.prelude(torch.tensor(ids)),
        [request.request_id] * len(ids),
        [0] * len(ids),
        list(range(len(ids))),
        cache,
        compute_gate=False,
    )
    model.recurrent(
        hidden,
        [request.request_id] * len(ids),
        [1] * len(ids),
        list(range(len(ids))),
        cache,
        compute_gate=False,
    )
    hidden, _ = model.recurrent(
        model.prelude(torch.tensor([token])),
        [request.request_id],
        [0],
        [len(ids)],
        cache,
        compute_gate=False,
    )
    return hidden[0]


def _chunk_span(request, *, depth, count=None, retain_chunk=False, retain_last=True):
    count = len(request.prompt_token_ids) if count is None else count
    return StateSpan(
        request,
        0,
        count,
        depth,
        chunk_start=0,
        retain_chunk=retain_chunk,
        retain_last=retain_last,
        prefill_hidden_row_offset=0,
    )


def _prepare(runner, batch, view):
    prepared = runner.prepare_on_cpu(batch, active_rows=view)
    prepared = runner.prepare_on_gpu(prepared)
    return prepared


@torch.inference_mode()
def test_mixed_rows_gather_scatter_and_generated_gate_mapping():
    torch.manual_seed(123)
    config = tiny_ouro_config()
    model, cache, runner = _make(config)
    reference_cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
        layout="last_exited",
    )
    prompt = Request("p", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    generated = Request("g", [7, 8], SamplingParams(max_tokens=1))

    prompt_hidden = _seed_prompt(model, cache, prompt, inherit=(0, 3))
    _seed_prompt(model, reference_cache, prompt, inherit=(0, 3))
    generated_hidden = _seed_generated(model, cache, generated, 9)
    _seed_generated(model, reference_cache, generated, 9)
    generated.hidden_state = generated_hidden.detach()
    slot = runner._ensure_request_slot(prompt)
    runner.prefill_states[slot, : len(prompt.prompt_token_ids)] = prompt_hidden

    spans = (
        _chunk_span(prompt, depth=1, count=4, retain_chunk=True, retain_last=True),
        StateSpan(
            generated,
            2,
            1,
            1,
            retain_last=True,
            emit_gate=True,
        ),
    )
    batch = ExecutionBatch(Operation.RECURRENT, spans, 1)
    view = ActiveRows(
        gather=(4, 1, 2),
        positions=(2, 1, 2),
        num_logical_rows=batch.num_rows,
        split=1,
    )
    prepared = runner.prepare_on_cpu(batch, active_rows=view)
    cpu_metadata = prepared.forwards[0].forward.metadata[0]
    assert len(cpu_metadata.rows) == 3
    assert [row[2] for row in cpu_metadata.rows] == [2, 1, 2]
    assert cpu_metadata.position_ids.shape[0] == 3
    assert {request_id for request_id, _ in cpu_metadata.allocations} == {"p", "g"}
    prepared = runner.prepare_on_gpu(prepared)
    entry = prepared.forwards[0]
    assert entry.scatter_index.tolist() == [4, 1, 2]
    assert entry.logical_hidden.shape[0] == 5
    assert entry.forward.hidden.shape[0] == 3
    assert len(entry.forward.metadata) == 1
    assert [row[1] for row in entry.forward.metadata[0].rows] == [1, 1, 1]
    assert [row[2] for row in entry.forward.metadata[0].rows] == [2, 1, 2]

    values = runner._execute(batch, prepared)

    prompt_reference, _ = model.recurrent(
        prompt_hidden[[1, 2]],
        ["p", "p"],
        [1, 1],
        [1, 2],
        reference_cache,
        compute_gate=False,
    )
    generated_reference, generated_gate = model.recurrent(
        generated_hidden.unsqueeze(0),
        ["g"],
        [1],
        [2],
        reference_cache,
        compute_gate=True,
    )
    expected_hidden = prompt_hidden.clone()
    expected_hidden[1] = prompt_reference[0]
    expected_hidden[2] = prompt_reference[1]
    torch.testing.assert_close(runner.prefill_states[slot, :4], expected_hidden)
    torch.testing.assert_close(prompt.hidden_state, expected_hidden[3])
    torch.testing.assert_close(generated.hidden_state, generated_reference[0])
    torch.testing.assert_close(values[1], generated_gate[0].float().sigmoid())

    for request_id, length in (("p", 4), ("g", 3)):
        for layer in range(config.num_hidden_layers):
            actual_key, actual_value = cache.read(layer, request_id, depth=1, length=length)
            expected_key, expected_value = reference_cache.read(
                layer, request_id, depth=1, length=length
            )
            torch.testing.assert_close(actual_key, expected_key)
            torch.testing.assert_close(actual_value, expected_value)


@torch.inference_mode()
def test_empty_view_skips_model_and_keeps_logical_hidden():
    torch.manual_seed(123)
    config = tiny_ouro_config()
    model, cache, runner = _make(config)
    prompt = Request("p", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    prompt_hidden = _seed_prompt(model, cache, prompt)
    slot = runner._ensure_request_slot(prompt)
    runner.prefill_states[slot, :4] = prompt_hidden

    def fail(*args, **kwargs):
        raise AssertionError("empty view must not call the model")

    runner.runtime.forward = fail
    batch = ExecutionBatch(
        Operation.RECURRENT,
        (_chunk_span(prompt, depth=1, retain_chunk=True, retain_last=True),),
        1,
    )
    view = ActiveRows(gather=(), positions=(), num_logical_rows=4, split=0)
    prepared = runner.prepare_on_cpu(batch, active_rows=view)
    cpu_metadata = prepared.forwards[0].forward.metadata
    assert len(cpu_metadata) == 1
    assert len(cpu_metadata[0].rows) == 0
    assert cpu_metadata[0].position_ids.shape[0] == 0
    assert {request_id for request_id, _ in cpu_metadata[0].allocations} == {"p"}
    prepared = runner.prepare_on_gpu(prepared)
    entry = prepared.forwards[0]
    assert entry.scatter_index.numel() == 0
    assert entry.logical_hidden.shape[0] == 4
    assert entry.forward.hidden.shape[0] == 0

    assert runner._execute(batch, prepared) is None
    torch.testing.assert_close(runner.prefill_states[slot, :4], prompt_hidden)
    torch.testing.assert_close(prompt.hidden_state, prompt_hidden[3])
    assert not any(cache.token_written("p", position, 1) for position in range(4))


@torch.inference_mode()
def test_dense_and_none_inputs_keep_the_established_path():
    torch.manual_seed(123)
    config = tiny_ouro_config()
    model, cache, runner = _make(config)
    prompt = Request("p", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    prompt_hidden = _seed_prompt(model, cache, prompt)
    slot = runner._ensure_request_slot(prompt)
    runner.prefill_states[slot, :4] = prompt_hidden
    batch = ExecutionBatch(Operation.RECURRENT, (_chunk_span(prompt, depth=1),), 1)

    plain = runner.prepare_on_gpu(runner.prepare_on_cpu(batch))
    dense = runner.prepare_on_gpu(
        runner.prepare_on_cpu(
            batch,
            active_rows=ActiveRows(
                gather=(0, 1, 2, 3),
                positions=(0, 1, 2, 3),
                num_logical_rows=4,
                split=None,
            ),
        )
    )
    for prepared in (plain, dense):
        entry = prepared.forwards[0]
        assert entry.scatter_index is None
        assert entry.logical_hidden is None
        assert entry.forward.hidden.shape[0] == 4


@torch.inference_mode()
def test_prepared_metadata_stays_stale_after_request_id_reuse():
    torch.manual_seed(123)
    config = tiny_ouro_config()
    model, cache, runner = _make(config)
    prompt = Request("p", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    prompt_hidden = _seed_prompt(model, cache, prompt)
    slot = runner._ensure_request_slot(prompt)
    runner.prefill_states[slot, :4] = prompt_hidden
    batch = ExecutionBatch(Operation.RECURRENT, (_chunk_span(prompt, depth=1),), 1)
    view = ActiveRows(gather=(1, 3), positions=(1, 3), num_logical_rows=4, split=0)
    prepared = runner.prepare_on_cpu(batch, active_rows=view)

    cache.free("p")
    assert cache.allocate("p", 8)
    cache.key_cache.fill_(17)
    before = cache.key_cache.clone()
    prepared = runner.prepare_on_gpu(prepared)
    with pytest.raises(RuntimeError, match="stale"):
        runner._execute(batch, prepared)
    torch.testing.assert_close(cache.key_cache, before)


@torch.inference_mode()
def test_inactive_emit_gate_target_is_rejected():
    config = tiny_ouro_config()
    _, _, runner = _make(config)
    request = Request("g", [1], SamplingParams(max_tokens=1))
    batch = ExecutionBatch(
        Operation.RECURRENT,
        (
            StateSpan(
                request,
                1,
                1,
                0,
                retain_last=True,
                emit_gate=True,
            ),
        ),
        1,
    )
    view = ActiveRows(gather=(), positions=(), num_logical_rows=1, split=0)
    with pytest.raises(ValueError, match="gate target"):
        runner.prepare_on_cpu(batch, active_rows=view)
