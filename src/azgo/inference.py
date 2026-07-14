"""Deterministic coordination and accounting for batched inference."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Condition, Lock, Thread, current_thread
from typing import TYPE_CHECKING, Literal, Never, Self

import numpy as np

from .evaluator import EvaluationBatch, Evaluator
from .game import GameState

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class InferenceError(RuntimeError):
    """Raised when deterministic inference cannot satisfy its contract."""


class InferenceClosedError(InferenceError):
    """Raised when a closed or inactive inference object is used."""


class InferenceAbortedError(InferenceError):
    """Raised when a coordinator abort releases an inference caller."""


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    """Aggregate model-inference work completed successfully."""

    requests: int
    positions: int
    batches: int
    max_batch_size: int
    mean_batch_size: float

    def __post_init__(self) -> None:
        integer_fields = (
            ("requests", self.requests),
            ("positions", self.positions),
            ("batches", self.batches),
            ("max_batch_size", self.max_batch_size),
        )
        for name, value in integer_fields:
            if type(value) is not int or value < 0:
                raise InferenceError(f"{name} must be a nonnegative integer")
        if type(self.mean_batch_size) is not float or not isfinite(self.mean_batch_size):
            raise InferenceError("mean_batch_size must be a finite nonnegative float")
        if self.mean_batch_size < 0.0:
            raise InferenceError("mean_batch_size must be a finite nonnegative float")
        if self.batches == 0:
            if (
                self.requests != 0
                or self.positions != 0
                or self.max_batch_size != 0
                or self.mean_batch_size != 0.0
            ):
                raise InferenceError("empty inference metrics must contain only zeros")
            return
        if self.requests == 0 or self.requests > self.positions:
            raise InferenceError("completed requests must lie in [1, positions]")
        if self.batches > self.positions:
            raise InferenceError("completed batches cannot exceed positions")
        if self.max_batch_size == 0 or self.max_batch_size > self.positions:
            raise InferenceError("max_batch_size must lie in [1, positions]")
        if self.mean_batch_size != self.positions / self.batches:
            raise InferenceError("mean_batch_size must equal positions divided by batches")


@dataclass(slots=True)
class _Request:
    states: tuple[GameState, ...]
    result: EvaluationBatch | None = None
    complete: bool = False


def _empty_metrics() -> InferenceMetrics:
    return InferenceMetrics(
        requests=0,
        positions=0,
        batches=0,
        max_batch_size=0,
        mean_batch_size=0.0,
    )


def _validate_max_batch_size(max_batch_size: int) -> int:
    if type(max_batch_size) is not int or max_batch_size <= 0:
        raise InferenceError("max_batch_size must be a positive integer")
    return max_batch_size


def _normalize_states(states: Sequence[GameState]) -> tuple[tuple[GameState, ...], int]:
    try:
        normalized = tuple(states)
    except BaseException as exc:
        raise InferenceError("inference states must be a finite sequence") from exc
    if not normalized:
        raise InferenceError("inference requires at least one state")

    first = normalized[0]
    if not isinstance(first, GameState):
        raise InferenceError("inference states must contain only GameState instances")
    board_size = first.board_size
    action_size = first.action_size
    for state in normalized[1:]:
        if not isinstance(state, GameState):
            raise InferenceError("inference states must contain only GameState instances")
        if state.board_size != board_size or state.action_size != action_size:
            raise InferenceError(
                "all states in an inference request must have the same board and action sizes"
            )
    return normalized, action_size


def _normalize_evaluation(
    evaluation: object,
    *,
    batch_size: int,
    action_size: int,
) -> EvaluationBatch:
    if not isinstance(evaluation, EvaluationBatch):
        raise InferenceError("evaluator must return an EvaluationBatch")
    policy_logits = evaluation.policy_logits
    values = evaluation.values
    if not isinstance(policy_logits, np.ndarray) or not isinstance(values, np.ndarray):
        raise InferenceError("evaluation outputs must be NumPy arrays")
    if policy_logits.dtype != np.float32 or values.dtype != np.float32:
        raise InferenceError("evaluation outputs must have dtype float32")
    if policy_logits.shape != (batch_size, action_size):
        raise InferenceError(
            "policy logits must have shape "
            f"({batch_size}, {action_size}), got {policy_logits.shape}"
        )
    if values.shape != (batch_size,):
        raise InferenceError(
            f"values must have shape ({batch_size},), got {values.shape}"
        )
    if not np.isfinite(policy_logits).all() or not np.isfinite(values).all():
        raise InferenceError("evaluation outputs must contain only finite values")
    if not ((values >= -1.0) & (values <= 1.0)).all():
        raise InferenceError("evaluation values must lie in [-1, 1]")

    return EvaluationBatch(
        policy_logits=np.array(policy_logits, dtype=np.float32, order="C", copy=True),
        values=np.array(values, dtype=np.float32, order="C", copy=True),
    )


class CountingEvaluator:
    """Wrap a direct evaluator and count its successful synchronous calls."""

    def __init__(self, evaluator: Evaluator) -> None:
        if not isinstance(evaluator, Evaluator):
            raise InferenceError("evaluator must implement the Evaluator protocol")
        self._evaluator = evaluator
        self._lock = Lock()
        self._requests = 0
        self._positions = 0
        self._batches = 0
        self._maximum = 0

    @property
    def metrics(self) -> InferenceMetrics:
        """Return a consistent snapshot of completed inference work."""

        with self._lock:
            mean = self._positions / self._batches if self._batches else 0.0
            return InferenceMetrics(
                requests=self._requests,
                positions=self._positions,
                batches=self._batches,
                max_batch_size=self._maximum,
                mean_batch_size=mean,
            )

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        """Evaluate immediately and record one request and model batch."""

        normalized, action_size = _normalize_states(states)
        try:
            evaluation = self._evaluator.evaluate_batch(normalized)
        except BaseException as exc:
            raise InferenceError("underlying evaluator failed") from exc
        result = _normalize_evaluation(
            evaluation,
            batch_size=len(normalized),
            action_size=action_size,
        )
        with self._lock:
            self._requests += 1
            self._positions += len(normalized)
            self._batches += 1
            self._maximum = max(self._maximum, len(normalized))
        return result


class InferenceClient:
    """A synchronous evaluator endpoint registered with a coordinator."""

    def __init__(self, coordinator: DeterministicInferenceCoordinator, client_id: int) -> None:
        self._coordinator = coordinator
        self._client_id = client_id

    @property
    def client_id(self) -> int:
        """Return this endpoint's immutable registered identifier."""

        return self._client_id

    def __enter__(self) -> Self:
        self._coordinator._ensure_client_open(self._client_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc_type, traceback
        if exc_value is not None:
            self._coordinator.abort(exc_value)
        self.close()
        return False

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        """Submit one request and block until its deterministic barrier completes."""

        return self._coordinator._evaluate(self._client_id, states)

    def close(self) -> None:
        """Retire this client from all future inference barriers."""

        self._coordinator._close_client(self._client_id)


class DeterministicInferenceCoordinator:
    """Batch registered clients at deterministic all-active-client barriers."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        max_batch_size: int,
        client_ids: Iterable[int],
    ) -> None:
        if not isinstance(evaluator, Evaluator):
            raise InferenceError("evaluator must implement the Evaluator protocol")
        self._evaluator = evaluator
        self._max_batch_size = _validate_max_batch_size(max_batch_size)
        try:
            identifiers = tuple(client_ids)
        except BaseException as exc:
            raise InferenceError("client_ids must be a finite iterable") from exc
        if not identifiers:
            raise InferenceError("client_ids must contain at least one identifier")
        if any(type(client_id) is not int or client_id < 0 for client_id in identifiers):
            raise InferenceError("client identifiers must be nonnegative integers")
        if len(set(identifiers)) != len(identifiers):
            raise InferenceError("client identifiers must be unique")

        self._client_ids = tuple(sorted(identifiers))
        self._condition = Condition()
        self._state: Literal["new", "running", "aborted", "closed"] = "new"
        self._active = set(self._client_ids)
        self._retiring: set[int] = set()
        self._requests: dict[int, _Request] = {}
        self._processing = False
        self._abort_cause: BaseException | None = None
        self._thread: Thread | None = None
        self._request_count = 0
        self._position_count = 0
        self._batch_count = 0
        self._maximum_batch = 0

    def __enter__(self) -> Self:
        startup_error: BaseException | None = None
        started_thread: Thread | None = None
        with self._condition:
            if self._state != "new":
                raise InferenceClosedError("inference coordinator cannot be entered again")
            self._state = "running"
            thread: Thread | None = None
            try:
                thread = Thread(
                    target=self._worker_main,
                    name="azgo-deterministic-inference",
                    daemon=False,
                )
                self._thread = thread
                thread.start()
            except BaseException as exc:
                startup_error = exc
                if thread is not None and thread.ident is not None:
                    started_thread = thread
                else:
                    self._thread = None
                self._abort_locked(exc)
        if startup_error is not None:
            if started_thread is not None and started_thread is not current_thread():
                started_thread.join()
            raise InferenceError("inference worker could not be started") from startup_error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc_type, traceback
        if exc_value is not None:
            self.abort(exc_value)
        self.close()
        return False

    @property
    def metrics(self) -> InferenceMetrics:
        """Return a consistent snapshot of completed barriers."""

        with self._condition:
            if self._batch_count == 0:
                return _empty_metrics()
            return InferenceMetrics(
                requests=self._request_count,
                positions=self._position_count,
                batches=self._batch_count,
                max_batch_size=self._maximum_batch,
                mean_batch_size=self._position_count / self._batch_count,
            )

    def client(self, client_id: int) -> InferenceClient:
        """Return an endpoint for one registered and active client."""

        if type(client_id) is not int:
            raise InferenceError("client identifier must be a nonnegative integer")
        with self._condition:
            if client_id not in self._client_ids:
                raise InferenceError(f"client {client_id} is not registered")
            if self._state in {"aborted", "closed"} or client_id not in self._active:
                raise InferenceClosedError(f"client {client_id} is closed")
        return InferenceClient(self, client_id)

    def abort(self, cause: BaseException) -> None:
        """Abort the coordinator and release every blocked client."""

        if not isinstance(cause, BaseException):
            raise InferenceError("abort cause must be a BaseException")
        with self._condition:
            self._abort_locked(cause)

    def close(self) -> None:
        """Stop inference, release callers, and join the dedicated worker."""

        thread: Thread | None
        with self._condition:
            if self._state == "new":
                self._state = "closed"
            elif self._state == "running":
                if self._requests:
                    self._abort_locked(
                        InferenceClosedError("coordinator closed with pending requests")
                    )
                else:
                    self._state = "closed"
                    self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._condition:
            if self._state != "aborted":
                self._state = "closed"
            self._condition.notify_all()

    def _ensure_client_open(self, client_id: int) -> None:
        with self._condition:
            if self._state != "running" or client_id not in self._active:
                raise InferenceClosedError(f"client {client_id} is not active")

    def _close_client(self, client_id: int) -> None:
        with self._condition:
            if client_id not in self._client_ids or client_id not in self._active:
                return
            if client_id in self._requests:
                self._retiring.add(client_id)
            else:
                self._active.remove(client_id)
            self._condition.notify_all()

    def _evaluate(
        self,
        client_id: int,
        states: Sequence[GameState],
    ) -> EvaluationBatch:
        with self._condition:
            self._check_submission_locked(client_id)
        try:
            normalized, _ = _normalize_states(states)
        except BaseException as exc:
            self.abort(exc)
            self._raise_aborted(exc)

        request = _Request(normalized)
        with self._condition:
            self._check_submission_locked(client_id)
            if client_id in self._requests:
                cause = InferenceError(f"client {client_id} already has an outstanding request")
                self._abort_locked(cause)
                self._raise_aborted(cause)
            self._requests[client_id] = request
            self._condition.notify_all()
            while not request.complete and self._state == "running":
                self._condition.wait()
            if request.complete and request.result is not None:
                return request.result
            if self._state == "aborted":
                self._raise_aborted(self._abort_cause)
            raise InferenceClosedError("inference coordinator closed before completing request")

    def _check_submission_locked(self, client_id: int) -> None:
        if self._state == "aborted":
            self._raise_aborted(self._abort_cause)
        if self._state != "running":
            raise InferenceClosedError("inference coordinator is not running")
        if client_id not in self._active or client_id in self._retiring:
            cause = InferenceClosedError(f"client {client_id} is inactive")
            self._abort_locked(cause)
            self._raise_aborted(cause)

    def _worker_main(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._state == "running" and not self._barrier_ready_locked():
                        if not self._active:
                            return
                        self._condition.wait()
                    if self._state != "running":
                        return
                    ordered_ids = sorted(self._active)
                    requests = [(client_id, self._requests[client_id]) for client_id in ordered_ids]
                    self._processing = True

                results, positions, batches, maximum = self._evaluate_barrier(requests)

                with self._condition:
                    if not self._is_running_locked():
                        self._processing = False
                        self._condition.notify_all()
                        return
                    for client_id, request in requests:
                        request.result = results[client_id]
                        request.complete = True
                        self._requests.pop(client_id, None)
                        if client_id in self._retiring:
                            self._retiring.remove(client_id)
                            self._active.discard(client_id)
                    self._request_count += len(requests)
                    self._position_count += positions
                    self._batch_count += batches
                    self._maximum_batch = max(self._maximum_batch, maximum)
                    self._processing = False
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._processing = False
                self._abort_locked(exc)

    def _barrier_ready_locked(self) -> bool:
        return (
            bool(self._active)
            and not self._processing
            and set(self._requests) == self._active
        )

    def _is_running_locked(self) -> bool:
        """Re-read lifecycle state after work performed outside the condition."""

        return self._state == "running"

    def _evaluate_barrier(
        self,
        requests: list[tuple[int, _Request]],
    ) -> tuple[dict[int, EvaluationBatch], int, int, int]:
        flattened: list[GameState] = []
        slices: dict[int, tuple[int, int]] = {}
        action_size: int | None = None
        board_size: int | None = None
        for client_id, request in requests:
            start = len(flattened)
            for state in request.states:
                if action_size is None:
                    action_size = state.action_size
                    board_size = state.board_size
                elif state.action_size != action_size or state.board_size != board_size:
                    raise InferenceError(
                        "all states in an inference barrier must have the same board and "
                        "action sizes"
                    )
                flattened.append(state)
            slices[client_id] = (start, len(flattened))

        if action_size is None:
            raise InferenceError("an inference barrier cannot be empty")
        policy_logits = np.empty((len(flattened), action_size), dtype=np.float32)
        values = np.empty(len(flattened), dtype=np.float32)
        batch_count = 0
        maximum = 0
        for start in range(0, len(flattened), self._max_batch_size):
            stop = min(start + self._max_batch_size, len(flattened))
            chunk = flattened[start:stop]
            try:
                raw = self._evaluator.evaluate_batch(chunk)
            except BaseException as exc:
                raise InferenceError("underlying evaluator failed") from exc
            result = _normalize_evaluation(
                raw,
                batch_size=len(chunk),
                action_size=action_size,
            )
            policy_logits[start:stop] = result.policy_logits
            values[start:stop] = result.values
            batch_count += 1
            maximum = max(maximum, len(chunk))

        mapped: dict[int, EvaluationBatch] = {}
        for client_id, (start, stop) in slices.items():
            mapped[client_id] = EvaluationBatch(
                policy_logits=np.array(
                    policy_logits[start:stop], dtype=np.float32, order="C", copy=True
                ),
                values=np.array(values[start:stop], dtype=np.float32, order="C", copy=True),
            )
        return mapped, len(flattened), batch_count, maximum

    def _abort_locked(self, cause: BaseException) -> None:
        if self._state in {"aborted", "closed"}:
            return
        self._abort_cause = cause
        self._state = "aborted"
        self._condition.notify_all()

    @staticmethod
    def _raise_aborted(cause: BaseException | None) -> Never:
        error = InferenceAbortedError("deterministic inference coordinator aborted")
        if cause is None:
            raise error
        raise error from cause
