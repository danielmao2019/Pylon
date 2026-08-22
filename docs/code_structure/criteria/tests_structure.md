# criteria — tests structure

## 1. Tests structure trees

`tests/criteria/base_criterion/test_base_criterion_async_buffer.py`

```text
test_base_criterion_async_buffer.py
├── import pytest
├── import torch
├── import time
├── import threading
├── from typing import Dict, Any, Optional
├── from criteria.base_criterion import BaseCriterion
├── @pytest.fixture def sample_tensor()
│   ├── # Offers one scalar loss value for a test to buffer.
│   ├── impls one scalar tensor
│   └── return  # that tensor
├── def test_high_frequency_buffer_operations(dummy_criterion)
│   ├── # A hundred losses pushed back to back all reach the buffer, in order and detached onto CPU, so the async hand-off loses nothing under load.
│   ├── impls a hundred distinct scalar tensors
│   ├── for each of those tensors
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── calls dummy_criterion._buffer_queue.join  # wait for the worker to drain
│   ├── calls dummy_criterion.get_buffer          # -> the accumulated buffer
│   ├── assert the buffer holds one entry per added tensor
│   ├── for each buffered tensor, in order
│   │   └── assert its value matches the one added at that position
│   └── for each buffered tensor
│       ├── assert it lives on CPU
│       └── assert it carries no gradient
├── def test_concurrent_buffer_access(dummy_criterion)
│   ├── # Three threads adding at once still yield exactly the union of their values, so the lock keeps concurrent producers from corrupting the buffer.
│   ├── def add_tensors_worker(start_idx: int, count: int) [local]
│   │   ├── # Pushes one thread's own contiguous slice of the value range, so the union across threads is checkable.
│   │   └── for each of this worker's count values
│   │       └── calls dummy_criterion.add_to_buffer
│   ├── calls add_tensors_worker  # reached indirectly, on each of the worker threads the loop below starts
│   ├── for each of three worker ids
│   │   ├── impls that worker's start index, ten values on from its predecessor's
│   │   ├── impls one thread targeting add_tensors_worker over that worker's slice
│   │   └── impls collect the thread
│   ├── for each thread
│   │   └── calls thread.start
│   ├── for each thread
│   │   └── calls thread.join
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds one entry per value added across all threads
│   └── assert the buffered values are the expected value range, ignoring order
├── def test_memory_pressure_queue_growth(dummy_criterion)
│   ├── # Fifty graph-carrying values pushed back to back all arrive detached on CPU, so a backlog costs latency rather than data or gradient state.
│   ├── for each of fifty values
│   │   ├── impls one scalar tensor that requires grad
│   │   ├── impls a scalar derived from it, so the value still carries its computation graph
│   │   └── impls collect that derived scalar
│   ├── for each of those tensors
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── assert the buffer queue still has items waiting  # timing-dependent: the daemon may drain all fifty before this reads qsize
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds one entry per added tensor
│   └── for each buffered tensor
│       ├── assert it lives on CPU
│       └── assert it carries no gradient
├── def test_queue_synchronization_join_behavior(dummy_criterion)
│   ├── # join returning is the signal that every queued value has been processed, which is what the rest of the suite synchronizes on.
│   ├── for each of twenty scalar values
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds one entry per added value
│   └── assert the buffer queue is empty once join returns
├── def test_buffer_thread_lifecycle(dummy_criterion)
│   ├── # The worker outlives the values it drains and stays a daemon, so it neither dies mid-run nor holds the process open at exit.
│   ├── assert the worker thread is alive right after construction
│   ├── for each of five scalar values
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── assert the worker thread is still alive after processing
│   └── assert the worker thread is a daemon
├── def test_worker_thread_error_propagation_detach()
│   ├── # A worker raising at the detach stage is left unhandled; the suite only checks the thread attribute survives, since observing the crash itself would need process isolation.
│   ├── calls FailingCriterion  # failing at the detach stage -> the criterion under test
│   ├── impls one scalar tensor the worker will fail on
│   ├── calls failing_criterion.add_to_buffer
│   ├── calls time.sleep(0.1)
│   ├── assert the criterion still exposes its worker thread attribute
│   └── assert that attribute is a threading.Thread
├── def test_buffer_state_consistency_during_operations(dummy_criterion)
│   ├── # Values read back in the order they were added, and a reset empties the buffer, so the trajectory a summary reduces is the one recorded.
│   ├── impls five scalar tensors of consecutive values
│   ├── for each of those tensors
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds all five
│   ├── for each buffered tensor, in order
│   │   └── assert its value matches the one added at that position
│   ├── calls dummy_criterion.reset_buffer
│   ├── calls dummy_criterion.get_buffer  # -> the buffer after the reset
│   └── assert that buffer is empty
├── def test_disabled_buffer_operations()
│   ├── # With buffering off no buffer state is built at all, adds are no-ops, and reading raises, so the disabled path stays honest about being off.
│   ├── calls DummyCriterion(use_buffer=False)
│   ├── assert buffering reads as disabled
│   ├── assert no buffer attribute was created
│   ├── assert no worker thread attribute was created
│   ├── impls one scalar tensor
│   ├── calls criterion.add_to_buffer  # accepted silently rather than raising
│   ├── assert still no buffer attribute was created
│   └── with pytest.raises(RuntimeError)  # matching the "Buffer is not enabled" message
│       └── calls criterion.get_buffer
├── def test_buffer_with_different_tensor_types(dummy_criterion)
│   ├── # A loss of any floating dtype lands on CPU, so the buffer is indifferent to the precision the criterion computed in.
│   ├── impls one scalar tensor per floating dtype under test — float32, float64, float16
│   ├── for each of those tensors
│   │   └── calls dummy_criterion.add_to_buffer
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds one entry per dtype
│   └── for each buffered tensor
│       └── assert it lives on CPU
├── @pytest.mark.parametrize def test_parametrized_concurrent_operations(dummy_criterion, num_workers, items_per_worker)  # over the (2, 5), (3, 10) and (4, 8) worker-count / items-per-worker pairs
│   ├── # The union-of-values guarantee holds across worker counts and per-worker loads, so it is the lock rather than one lucky schedule that carries it.
│   ├── def worker_function(worker_id: int) [local]
│   │   ├── # Pushes one worker's own contiguous slice of the value range, so the union across workers is checkable.
│   │   └── for each of this worker's items
│   │       └── calls dummy_criterion.add_to_buffer
│   ├── calls worker_function  # reached indirectly, on each of the worker threads the loop below starts
│   ├── for each worker id
│   │   ├── impls one thread targeting worker_function for that worker
│   │   ├── impls collect the thread
│   │   └── calls thread.start
│   ├── for each thread
│   │   └── calls thread.join
│   ├── calls dummy_criterion._buffer_queue.join
│   ├── calls dummy_criterion.get_buffer  # -> the accumulated buffer
│   ├── assert the buffer holds one entry per value added across all workers
│   └── assert the buffered values are the expected value range, ignoring order
├── def test_extreme_lock_contention_resilience()
│   ├── # Eight threads adding and reading at once neither deadlock nor lose a value, so the lock holds under more contention than a real run produces.
│   ├── calls DummyCriterion(use_buffer=True)
│   ├── def contention_worker(worker_id: int, iterations: int) [local]
│   │   ├── # Interleaves adds with periodic reads, so producers and readers contend for the same lock.
│   │   └── for each of this worker's iterations
│   │       ├── calls criterion.add_to_buffer
│   │       └── if this is every fifth iteration
│   │           ├── try
│   │           │   ├── calls criterion.get_buffer  # takes the lock mid-contention
│   │           │   └── assert what comes back is a list
│   │           └── except Exception
│   │               └── pass  # some contention is expected, continue
│   ├── calls contention_worker  # reached indirectly, on each of the worker threads the loop below starts
│   ├── for each of eight worker ids
│   │   ├── impls one thread targeting contention_worker over twenty-five iterations
│   │   └── impls collect the thread
│   ├── impls take the start timestamp
│   ├── for each thread
│   │   └── calls thread.start
│   ├── for each thread
│   │   └── calls thread.join
│   ├── impls take the elapsed contention time
│   ├── calls criterion._buffer_queue.join
│   ├── calls criterion.get_buffer                # -> the accumulated buffer
│   ├── assert no data was lost under contention  # "Lost data under contention", reporting the buffered and expected counts
│   ├── assert the elapsed time stayed under the thirty-second deadlock bound    # "Potential deadlock detected", reporting the elapsed seconds
│   └── assert the buffered values are the expected value range, ignoring order  # "Data corruption under extreme contention"
├── def test_check_validity_parameter()
│   ├── # check_validity is what decides whether an inf or NaN loss is rejected at the door or buffered anyway.
│   ├── calls DummyCriterion(use_buffer=True)
│   ├── impls invalid_tensor — one infinite scalar tensor
│   ├── with pytest.raises(AssertionError)  # matching the offending-value message
│   │   └── calls criterion.add_to_buffer(invalid_tensor, check_validity=True)
│   ├── calls criterion.add_to_buffer(invalid_tensor, check_validity=False)
│   ├── impls nan_tensor — one NaN scalar tensor
│   ├── with pytest.raises(AssertionError)  # matching the offending-value message
│   │   └── calls criterion.add_to_buffer(nan_tensor, check_validity=True)
│   ├── calls criterion.add_to_buffer(nan_tensor, check_validity=False)
│   ├── calls criterion._buffer_queue.join
│   ├── calls criterion.get_buffer  # -> the accumulated buffer
│   └── assert both unchecked values were buffered
├── def test_invalid_tensor_shape_assertions()
│   ├── # Only a single-element zero-dimensional tensor is buffered, so a whole loss map can never be mistaken for one scalar loss.
│   ├── calls DummyCriterion(use_buffer=True)
│   ├── impls one multi-dimensional tensor
│   ├── with pytest.raises(AssertionError)  # matching the offending-shape message
│   │   └── calls criterion.add_to_buffer  # the multi-dimensional tensor
│   ├── impls one multi-element tensor
│   ├── with pytest.raises(AssertionError)  # matching the offending-shape message
│   │   └── calls criterion.add_to_buffer  # the multi-element tensor
│   ├── impls one empty tensor
│   ├── with pytest.raises(AssertionError)  # matching the offending-shape message
│   │   └── calls criterion.add_to_buffer  # the empty tensor
│   ├── impls one scalar tensor
│   ├── calls criterion.add_to_buffer(valid_tensor)
│   ├── calls criterion._buffer_queue.join
│   ├── calls criterion.get_buffer  # -> the accumulated buffer
│   └── assert only the scalar tensor was buffered
├── def test_invalid_tensor_type_assertions()
│   ├── # Only a torch.Tensor is buffered, so a Python scalar or None cannot enter the trajectory a summary later stacks.
│   ├── calls DummyCriterion(use_buffer=True)
│   ├── with pytest.raises(AssertionError)  # matching the offending-type message
│   │   └── calls criterion.add_to_buffer  # a plain float
│   ├── with pytest.raises(AssertionError)  # matching the offending-type message
│   │   └── calls criterion.add_to_buffer  # a list
│   ├── with pytest.raises(AssertionError)  # matching the offending-type message
│   │   └── calls criterion.add_to_buffer  # a string
│   └── with pytest.raises(AssertionError)  # matching the offending-type message
│       └── calls criterion.add_to_buffer(None)
├── def test_empty_buffer_edge_cases()
│   ├── # Reading and resetting an already-empty buffer are both safe and repeatable, so the empty state is a real state rather than an error.
│   ├── calls DummyCriterion(use_buffer=True)
│   ├── calls criterion.get_buffer  # -> the buffer before anything is added
│   ├── assert that buffer is empty
│   ├── assert that buffer is a list
│   ├── calls criterion.reset_buffer  # resetting an already-empty buffer
│   ├── assert the buffer is still empty
│   ├── calls criterion.reset_buffer
│   ├── calls criterion.reset_buffer
│   ├── assert repeated resets leave the buffer empty
│   ├── calls criterion.add_to_buffer(torch.tensor(1.0))
│   ├── calls criterion._buffer_queue.join
│   ├── assert the buffer holds that one value
│   ├── calls criterion.reset_buffer
│   ├── assert resetting after data leaves the buffer empty
│   ├── for each of three scalar values
│   │   └── calls criterion.add_to_buffer
│   ├── calls criterion._buffer_queue.join
│   ├── assert the buffer holds all three
│   ├── calls criterion.reset_buffer
│   ├── assert the buffer is empty again
│   └── assert the buffer queue is empty after the reset
├── class FailingCriterion(BaseCriterion)
│   ├── # Same loss as DummyCriterion, with a worker loop that raises at whichever stage fail_mode names.
│   ├── def __init__(self, fail_mode: str = "detach")  [override]
│   │   ├── # Turns buffering on and records which stage of the worker loop the injected failure fires at.
│   │   ├── calls super().__init__  # with buffering on
│   │   └── impls remember the failure mode
│   ├── def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor  [override]
│   │   ├── # Scores the same mean absolute error as DummyCriterion, so the failure under test is the worker's alone.
│   │   ├── impls take the absolute difference between prediction and ground truth  # impls-node-one-step:skip
│   │   └── return  # the mean of that difference
│   ├── def summarize(self, output_path: Optional[str] = None) -> torch.Tensor  [override]
│   │   ├── # Satisfies the abstract contract with a constant zero scalar.
│   │   ├── impls one zero scalar tensor
│   │   └── return  # that constant, standing in for a real summary
│   └── def _buffer_worker(self) -> None  [override]
│       ├── # Reproduces the real worker loop but raises at whichever stage fail_mode names, so the unhandled-failure path is exercised.
│       └── while True
│           ├── impls take the next value off _buffer_queue
│           ├── if the failure mode is the detach one
│           │   └── raise RuntimeError  # "Simulated detach failure"
│           ├── elif the failure mode is the CPU-transfer one
│           │   ├── impls detach the value, reaching the point the transfer would follow
│           │   └── raise RuntimeError  # "Simulated CPU transfer failure"
│           ├── elif the failure mode is the lock one
│           │   └── raise RuntimeError  # "Simulated lock failure"
│           ├── with _buffer_lock
│           │   ├── impls detach that value onto CPU
│           │   └── impls append it to buffer
│           └── impls mark the _buffer_queue task done
├── @pytest.fixture def dummy_criterion()
│   ├── # Gives each test its own buffering criterion, so no test inherits another's accumulated trajectory.
│   ├── calls DummyCriterion(use_buffer=True)
│   └── return  # that criterion, fresh for each test that asks for it
└── class DummyCriterion(BaseCriterion)
    ├── # The concrete criterion the suite buffers through: a mean-absolute-error loss and a mean-of-buffer summary.
    ├── def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor  [override]
    │   ├── # Gives the tests a real scalar loss to buffer, as the mean absolute error between the two tensors.
    │   ├── impls take the absolute difference between prediction and ground truth  # impls-node-one-step:skip
    │   └── return  # the mean of that difference
    └── def summarize(self, output_path: Optional[str] = None) -> torch.Tensor  [override]
        ├── # Reduces the buffered losses to their mean, saving it to output_path when one is given.
        ├── assert buffering is enabled     # "Buffer must be enabled"
        ├── assert the buffer is non-empty  # "Buffer must not be empty"
        ├── impls take the mean over the stacked buffered losses
        ├── if an output path was given
        │   └── calls torch.save(result, output_path)
        └── return  # the mean of the buffered losses
```
