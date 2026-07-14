"""Tests for deterministic concurrent inference coordination."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import numpy as np
import pytest

from azgo.evaluator import EvaluationBatch, Evaluator, UniformEvaluator
from azgo.game import GameState
from azgo.inference import (
    CountingEvaluator,
    DeterministicInferenceCoordinator,
    InferenceAbortedError,
    InferenceClosedError,
    InferenceError,
    InferenceMetrics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from azgo.inference import InferenceClient


def _state_at(move_number: int, board_size: int = 5) -> GameState:
    state = GameState.new(board_size)
    for _ in range(move_number):
        action = next(action for action in state.legal_actions() if action != state.pass_action)
        state = state.apply(action)
    return state


class TraceEvaluator:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.trace: list[tuple[int, ...]] = []
        self.thread_ids: list[int] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        self.started.set()
        if self.block and not self.release.wait(2.0):
            raise RuntimeError("test evaluator release timed out")
        if self.delay:
            time.sleep(self.delay)
        self.trace.append(tuple(state.move_number for state in states))
        self.thread_ids.append(threading.get_ident())
        logits = np.zeros((len(states), states[0].action_size), dtype=np.float32)
        values = np.empty(len(states), dtype=np.float32)
        for index, state in enumerate(states):
            logits[index, 0] = np.float32(state.move_number)
            values[index] = np.float32(state.move_number / 100.0)
        return EvaluationBatch(logits, values)


class FailingEvaluator:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        del states
        raise self.error


class ResultEvaluator:
    def __init__(self, result: object) -> None:
        self.result = result

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        del states
        return self.result  # type: ignore[return-value]


def _start_call(
    client: InferenceClient,
    states: Sequence[GameState],
    *,
    delay: float = 0.0,
) -> tuple[threading.Thread, list[EvaluationBatch], list[BaseException]]:
    results: list[EvaluationBatch] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            if delay:
                time.sleep(delay)
            results.append(client.evaluate_batch(states))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, results, errors


def _join(thread: threading.Thread) -> None:
    thread.join(3.0)
    assert not thread.is_alive(), "inference caller was stranded"


def test_public_evaluators_conform_to_protocol() -> None:
    assert isinstance(CountingEvaluator(UniformEvaluator()), Evaluator)
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0]
    ) as coordinator:
        assert isinstance(coordinator.client(0), Evaluator)


def test_empty_metrics_and_fields_are_frozen() -> None:
    metrics = InferenceMetrics(0, 0, 0, 0, 0.0)

    assert metrics == InferenceMetrics(0, 0, 0, 0, 0.0)
    with pytest.raises(FrozenInstanceError):
        metrics.requests = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "metrics",
    [
        InferenceMetrics(0, 0, 0, 0, 0.0),
        InferenceMetrics(1, 1, 1, 1, 1.0),
        InferenceMetrics(2, 3, 2, 2, 1.5),
    ],
)
def test_valid_metric_combinations(metrics: InferenceMetrics) -> None:
    assert metrics.mean_batch_size >= 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        (-1, 0, 0, 0, 0.0),
        (True, 1, 1, 1, 1.0),
        (1, 0, 0, 0, 0.0),
        (0, 1, 0, 0, 0.0),
        (0, 0, 1, 0, 0.0),
        (1, 1, 1, 0, 1.0),
        (2, 1, 1, 1, 1.0),
        (1, 1, 2, 1, 0.5),
        (1, 1, 1, 2, 1.0),
        (1, 2, 1, 1, 1.0),
        (1, 1, 1, 1, float("nan")),
        (1, 1, 1, 1, -1.0),
        (1, 1, 1, 1, 1),
    ],
)
def test_metrics_reject_inconsistent_or_non_strict_fields(arguments: tuple[object, ...]) -> None:
    with pytest.raises(InferenceError):
        InferenceMetrics(*arguments)  # type: ignore[arg-type]


def test_counting_evaluator_returns_detached_contiguous_results_and_metrics() -> None:
    policy = np.zeros((2, 26), dtype=np.float32)
    values = np.array([0.25, -0.25], dtype=np.float32)
    direct = CountingEvaluator(ResultEvaluator(EvaluationBatch(policy, values)))

    result = direct.evaluate_batch([_state_at(0), _state_at(1)])
    policy[0, 0] = 99.0
    values[0] = 1.0

    assert result.policy_logits[0, 0] == 0.0
    assert result.values[0] == 0.25
    assert result.policy_logits.flags.c_contiguous
    assert result.values.flags.c_contiguous
    assert direct.metrics == InferenceMetrics(1, 2, 1, 2, 2.0)


def test_counting_evaluator_accumulates_one_batch_per_successful_request() -> None:
    direct = CountingEvaluator(UniformEvaluator())

    direct.evaluate_batch([_state_at(0)])
    direct.evaluate_batch([_state_at(1), _state_at(2), _state_at(3)])

    assert direct.metrics == InferenceMetrics(2, 4, 2, 3, 2.0)


@pytest.mark.parametrize(
    "result",
    [
        object(),
        EvaluationBatch(np.zeros((1, 25), dtype=np.float32), np.zeros(1, dtype=np.float32)),
        EvaluationBatch(np.zeros((1, 26), dtype=np.float64), np.zeros(1, dtype=np.float32)),
        EvaluationBatch(np.zeros((1, 26), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)),
        EvaluationBatch(
            np.full((1, 26), np.nan, dtype=np.float32), np.zeros(1, dtype=np.float32)
        ),
        EvaluationBatch(np.zeros((1, 26), dtype=np.float32), np.array([1.01], np.float32)),
    ],
)
def test_counting_evaluator_rejects_bad_output_without_counting(result: object) -> None:
    direct = CountingEvaluator(ResultEvaluator(result))

    with pytest.raises(InferenceError):
        direct.evaluate_batch([_state_at(0)])

    assert direct.metrics == InferenceMetrics(0, 0, 0, 0, 0.0)


def test_counting_evaluator_wraps_underlying_failure() -> None:
    direct = CountingEvaluator(FailingEvaluator(ValueError("broken")))

    with pytest.raises(InferenceError, match="underlying evaluator failed") as error:
        direct.evaluate_batch([_state_at(0)])

    assert isinstance(error.value.__cause__, ValueError)
    assert direct.metrics == InferenceMetrics(0, 0, 0, 0, 0.0)


@pytest.mark.parametrize("max_batch_size", [0, -1, True, 1.5])
def test_coordinator_rejects_invalid_max_batch_size(max_batch_size: object) -> None:
    with pytest.raises(InferenceError, match="positive integer"):
        DeterministicInferenceCoordinator(
            UniformEvaluator(),
            max_batch_size=max_batch_size,  # type: ignore[arg-type]
            client_ids=[0],
        )


@pytest.mark.parametrize("client_ids", [[], [0, 0], [-1], [True], [1.5]])
def test_coordinator_rejects_invalid_registered_clients(client_ids: list[object]) -> None:
    with pytest.raises(InferenceError):
        DeterministicInferenceCoordinator(
            UniformEvaluator(), max_batch_size=1, client_ids=client_ids  # type: ignore[arg-type]
        )


def test_barrier_sorts_clients_chunks_positions_and_maps_results() -> None:
    evaluator = TraceEvaluator()
    main_thread = threading.get_ident()
    with DeterministicInferenceCoordinator(
        evaluator, max_batch_size=2, client_ids=[8, 4, 1]
    ) as coordinator:
        clients = {client_id: coordinator.client(client_id) for client_id in (8, 4, 1)}
        calls = [
            _start_call(clients[8], [_state_at(3)], delay=0.0),
            _start_call(clients[4], [_state_at(4)], delay=0.01),
            _start_call(clients[1], [_state_at(1), _state_at(2)], delay=0.02),
        ]
        for thread, _, _ in calls:
            _join(thread)

        assert all(not errors for _, _, errors in calls)
        by_client = {8: calls[0][1][0], 4: calls[1][1][0], 1: calls[2][1][0]}
        assert by_client[1].policy_logits[:, 0].tolist() == [1.0, 2.0]
        assert by_client[4].policy_logits[:, 0].tolist() == [4.0]
        assert by_client[8].policy_logits[:, 0].tolist() == [3.0]
        assert all(result.policy_logits.flags.c_contiguous for result in by_client.values())
        assert evaluator.trace == [(1, 2), (4, 3)]
        assert evaluator.thread_ids
        assert set(evaluator.thread_ids) != {main_thread}
        assert len(set(evaluator.thread_ids)) == 1
        assert coordinator.metrics == InferenceMetrics(3, 4, 2, 2, 2.0)


def test_artificial_submission_delays_produce_repeated_deterministic_traces() -> None:
    traces: list[list[tuple[int, ...]]] = []
    for delays in ((0.03, 0.01, 0.0), (0.0, 0.02, 0.04)):
        evaluator = TraceEvaluator()
        with DeterministicInferenceCoordinator(
            evaluator, max_batch_size=2, client_ids=[0, 1, 2]
        ) as coordinator:
            calls = [
                _start_call(coordinator.client(index), [_state_at(index + 1)], delay=delay)
                for index, delay in enumerate(delays)
            ]
            for thread, _, _ in calls:
                _join(thread)
        traces.append(evaluator.trace)

    assert traces == [[(1, 2), (3,)], [(1, 2), (3,)]]


def test_client_completion_removes_it_from_future_barriers() -> None:
    evaluator = TraceEvaluator()
    with DeterministicInferenceCoordinator(
        evaluator, max_batch_size=8, client_ids=[0, 1, 2]
    ) as coordinator:
        client0 = coordinator.client(0)
        client1 = coordinator.client(1)
        coordinator.client(2).close()
        first = _start_call(client0, [_state_at(1)])
        second = _start_call(client1, [_state_at(2)])
        _join(first[0])
        _join(second[0])
        client1.close()
        third = _start_call(client0, [_state_at(3)])
        _join(third[0])

        assert not first[2]
        assert not second[2]
        assert not third[2]
        assert evaluator.trace == [(1, 2), (3,)]
        assert coordinator.metrics == InferenceMetrics(3, 3, 2, 2, 1.5)


def test_closing_client_with_pending_request_retires_after_current_barrier() -> None:
    evaluator = TraceEvaluator()
    with DeterministicInferenceCoordinator(
        evaluator, max_batch_size=4, client_ids=[0, 1]
    ) as coordinator:
        client0 = coordinator.client(0)
        client1 = coordinator.client(1)
        first = _start_call(client0, [_state_at(1)])
        time.sleep(0.02)
        client0.close()
        second = _start_call(client1, [_state_at(2)])
        _join(first[0])
        _join(second[0])

        assert not first[2]
        assert not second[2]
        with pytest.raises(InferenceClosedError):
            coordinator.client(0)


def test_invalid_request_aborts_and_releases_every_waiter() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        waiting = _start_call(coordinator.client(0), [_state_at(0)])
        time.sleep(0.02)

        with pytest.raises(InferenceAbortedError) as invalid:
            coordinator.client(1).evaluate_batch([])
        _join(waiting[0])

        assert isinstance(invalid.value.__cause__, InferenceError)
        assert len(waiting[2]) == 1
        assert isinstance(waiting[2][0], InferenceAbortedError)


def test_duplicate_submission_aborts_both_calls() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        client = coordinator.client(0)
        first = _start_call(client, [_state_at(0)])
        time.sleep(0.02)
        duplicate = _start_call(client, [_state_at(1)])
        _join(first[0])
        _join(duplicate[0])

        assert isinstance(first[2][0], InferenceAbortedError)
        assert isinstance(duplicate[2][0], InferenceAbortedError)


@pytest.mark.parametrize(
    "result",
    [
        object(),
        EvaluationBatch(np.zeros((1, 25), np.float32), np.zeros(1, np.float32)),
        EvaluationBatch(np.zeros((1, 26), np.float64), np.zeros(1, np.float32)),
        EvaluationBatch(np.zeros((1, 26), np.float32), np.array([np.inf], np.float32)),
        EvaluationBatch(np.zeros((1, 26), np.float32), np.array([-1.1], np.float32)),
    ],
)
def test_malformed_worker_output_aborts_all_clients(result: object) -> None:
    with DeterministicInferenceCoordinator(
        ResultEvaluator(result), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        calls = [
            _start_call(coordinator.client(0), [_state_at(0)]),
            _start_call(coordinator.client(1), [_state_at(1)]),
        ]
        for thread, _, _ in calls:
            _join(thread)

        assert all(isinstance(errors[0], InferenceAbortedError) for _, _, errors in calls)
        assert coordinator.metrics == InferenceMetrics(0, 0, 0, 0, 0.0)


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_worker_catches_base_exception_and_releases_all_clients(failure: BaseException) -> None:
    with DeterministicInferenceCoordinator(
        FailingEvaluator(failure), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        calls = [
            _start_call(coordinator.client(0), [_state_at(0)]),
            _start_call(coordinator.client(1), [_state_at(1)]),
        ]
        for thread, _, _ in calls:
            _join(thread)

        assert all(isinstance(errors[0], InferenceAbortedError) for _, _, errors in calls)


def test_mixed_board_sizes_between_clients_abort_barrier() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        calls = [
            _start_call(coordinator.client(0), [GameState.new(5)]),
            _start_call(coordinator.client(1), [GameState.new(9)]),
        ]
        for thread, _, _ in calls:
            _join(thread)

        assert all(isinstance(errors[0], InferenceAbortedError) for _, _, errors in calls)


def test_explicit_abort_during_inflight_evaluation_discards_results_and_metrics() -> None:
    evaluator = TraceEvaluator()
    evaluator.block = True
    with DeterministicInferenceCoordinator(
        evaluator, max_batch_size=2, client_ids=[0]
    ) as coordinator:
        call = _start_call(coordinator.client(0), [_state_at(1)])
        assert evaluator.started.wait(1.0)
        cause = RuntimeError("cancelled")
        coordinator.abort(cause)
        evaluator.release.set()
        _join(call[0])

        assert isinstance(call[2][0], InferenceAbortedError)
        assert call[2][0].__cause__ is cause
        assert not call[1]
        assert coordinator.metrics == InferenceMetrics(0, 0, 0, 0, 0.0)


def test_close_with_pending_request_releases_waiter_and_joins_worker() -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0, 1]
    )
    coordinator.__enter__()
    call = _start_call(coordinator.client(0), [_state_at(0)])
    time.sleep(0.02)

    coordinator.close()
    _join(call[0])

    assert isinstance(call[2][0], InferenceAbortedError)
    with pytest.raises(InferenceClosedError):
        coordinator.client(0)
    assert not any(
        thread.name == "azgo-deterministic-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_calls_before_entry_and_after_normal_close_fail() -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    )
    client = coordinator.client(0)
    with pytest.raises(InferenceClosedError, match="not running"):
        client.evaluate_batch([_state_at(0)])

    with coordinator:
        result = client.evaluate_batch([_state_at(0)])
        assert result.values.tolist() == [0.0]

    with pytest.raises(InferenceClosedError):
        coordinator.client(0)
    with pytest.raises(InferenceClosedError):
        client.evaluate_batch([_state_at(0)])


def test_client_context_exception_aborts_peer_waiter() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=2, client_ids=[0, 1]
    ) as coordinator:
        peer = _start_call(coordinator.client(0), [_state_at(0)])
        time.sleep(0.02)
        with pytest.raises(ValueError, match="worker failed"), coordinator.client(1):
            raise ValueError("worker failed")
        _join(peer[0])

        assert isinstance(peer[2][0], InferenceAbortedError)


def test_client_lookup_rejects_unknown_and_non_integer_identifiers() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[2]
    ) as coordinator:
        with pytest.raises(InferenceError, match="not registered"):
            coordinator.client(1)
        with pytest.raises(InferenceError, match="nonnegative integer"):
            coordinator.client(True)


def test_coordinator_cannot_be_reentered() -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    )
    with coordinator, pytest.raises(InferenceClosedError, match="entered again"):
        coordinator.__enter__()


def test_worker_start_failure_rolls_back_lifecycle_and_never_joins_unstarted_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    )
    client = coordinator.client(0)
    cause = RuntimeError("thread start failed")

    def fail_start(thread: threading.Thread) -> None:
        del thread
        raise cause

    monkeypatch.setattr("azgo.inference.Thread.start", fail_start)

    with pytest.raises(InferenceError, match="could not be started") as error:
        coordinator.__enter__()
    assert error.value.__cause__ is cause
    with pytest.raises(InferenceAbortedError) as aborted:
        client.evaluate_batch([_state_at(0)])
    assert aborted.value.__cause__ is cause

    coordinator.close()
    coordinator.close()
    assert not any(
        thread.name == "azgo-deterministic-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_worker_constructor_failure_rolls_back_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    )
    client = coordinator.client(0)
    cause = RuntimeError("thread construction failed")

    def fail_construction(*args: object, **kwargs: object) -> threading.Thread:
        del args, kwargs
        raise cause

    monkeypatch.setattr("azgo.inference.Thread", fail_construction)

    with pytest.raises(InferenceError, match="could not be started") as error:
        coordinator.__enter__()
    assert error.value.__cause__ is cause
    with pytest.raises(InferenceAbortedError) as aborted:
        client.evaluate_batch([_state_at(0)])
    assert aborted.value.__cause__ is cause
    coordinator.close()


def test_start_that_raises_after_launch_is_aborted_and_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    )
    client = coordinator.client(0)
    original_start = threading.Thread.start
    cause = RuntimeError("start failed after launch")

    def launch_then_fail(thread: threading.Thread) -> None:
        original_start(thread)
        raise cause

    monkeypatch.setattr("azgo.inference.Thread.start", launch_then_fail)

    with pytest.raises(InferenceError, match="could not be started") as error:
        coordinator.__enter__()
    assert error.value.__cause__ is cause
    with pytest.raises(InferenceAbortedError) as aborted:
        client.evaluate_batch([_state_at(0)])
    assert aborted.value.__cause__ is cause
    coordinator.close()
    assert not any(
        thread.name == "azgo-deterministic-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_inference_client_identifier_is_read_only() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[7]
    ) as coordinator:
        client = coordinator.client(7)
        assert client.client_id == 7
        with pytest.raises(AttributeError):
            client.client_id = 8  # type: ignore[misc]


def test_abort_requires_base_exception() -> None:
    with DeterministicInferenceCoordinator(
        UniformEvaluator(), max_batch_size=1, client_ids=[0]
    ) as coordinator, pytest.raises(InferenceError, match=re.escape("BaseException")):
        coordinator.abort("stop")  # type: ignore[arg-type]
