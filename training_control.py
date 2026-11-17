"""Session-local, cooperative stopping for streamed GNN training."""

from functools import wraps
from html import escape
import inspect
import threading

import gradio as gr


class TrainingStopped(Exception):
    """A user requested a stop at a safe boundary between batches."""


class TrainingControl:
    """Share a stop signal between one session's training and Stop callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = False

    def __deepcopy__(self, memo):
        # Gradio deepcopies a State's initial value for each browser session.
        # Locks/events must never be copied or shared between those sessions.
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    @property
    def is_running(self):
        with self._lock:
            return self._running

    @property
    def stop_requested(self):
        return self._stop.is_set()

    def start(self):
        with self._lock:
            if self._running:
                raise ValueError("Training is already running in this tab.")
            self._stop.clear()
            self._running = True

    def finish(self):
        with self._lock:
            self._running = False
            self._stop.clear()

    def request_stop(self):
        with self._lock:
            if not self._running:
                return False
            self._stop.set()
            return True

    def check(self):
        if self._stop.is_set():
            raise TrainingStopped()


class InterruptibleLoader:
    """Check the stop flag before fetching and before computing each batch."""

    def __init__(self, loader, control):
        self.loader = loader
        self.control = control

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def check_stop(self):
        self.control.check()

    def __iter__(self):
        self.check_stop()
        iterator = iter(self.loader)
        try:
            while True:
                self.check_stop()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                self.check_stop()
                yield batch
        finally:
            # Our loaders use nonpersistent spawned workers. Explicit shutdown
            # also covers interrupted epochs whose iterator is not exhausted.
            # Generic iterators remain supported without PyTorch private APIs.
            try:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            finally:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown) and not getattr(self.loader, "persistent_workers", False):
                    shutdown()


def bind_training(button, callback, inputs, outputs, *, stop_button, control_state,
                  clear_states, disable_buttons, clear_components, concurrency_id=None):
    """Stream updates under the tab lock, with an independent Stop callback.

    ``callback.stream`` has the same signature as the synchronous handler and
    yields its ordinary outputs with a ``finished`` attribute. Keeping the
    synchronous handler available preserves direct callers and existing tests.
    """
    all_outputs = [*outputs, *clear_states, *disable_buttons, *clear_components,
                   button, stop_button]
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("Training outputs and result resets must not overlap.")

    signature = inspect.signature(callback)
    parameters = list(signature.parameters.values())
    progress_index = next((index for index, parameter in enumerate(parameters)
                           if isinstance(parameter.default, gr.Progress)), len(parameters))
    parameters.insert(progress_index, inspect.Parameter(
        "training_control", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None))
    streaming_signature = signature.replace(parameters=parameters)

    def clear_results(result):
        result.update((state, None) for state in clear_states)
        result.update((component, gr.update(interactive=False)) for component in disable_buttons)
        result.update((component, gr.update(value=None)) for component in clear_components)
        return result

    @wraps(callback)
    def stream_and_reset(*args, **kwargs):
        bound = streaming_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        control = bound.arguments.pop("training_control")
        owns_run = False
        stream = None
        try:
            if not isinstance(control, TrainingControl):
                raise ValueError("Training controls are unavailable. Please reload this tab.")
            control.start()
            owns_run = True
            for name in ("train_dataloader", "val_dataloader"):
                loader = bound.arguments.get(name)
                if loader is not None:
                    bound.arguments[name] = InterruptibleLoader(loader, control)
            stream = callback.stream(**bound.arguments)
            for update in stream:
                finished = getattr(update, "finished", True)
                result = clear_results(dict(zip(outputs, update, strict=True)))
                if finished:
                    control.finish()
                    owns_run = False
                stopping = control.stop_requested
                result[button] = gr.update(interactive=finished)
                result[stop_button] = gr.update(
                    value="Stopping…" if stopping else "Stop training",
                    interactive=not finished and not stopping)
                yield result
        except Exception as exc:
            if owns_run:
                control.finish()
                owns_run = False
                result = clear_results({outputs[0]: (
                    "<span style='color: red;'>Error during training: "
                    f"{escape(str(exc))}</span>")})
                result[button] = gr.update(interactive=True)
                result[stop_button] = gr.update(value="Stop training", interactive=False)
            else:
                # A duplicate API request must not take ownership of the active
                # generator or change its stop signal and button state.
                result = {outputs[0]: (
                    "<span style='color: red;'>Error during training: "
                    f"{escape(str(exc))}</span>")}
            yield result
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                if owns_run:
                    control.finish()

    stream_and_reset.__signature__ = streaming_signature

    def stop_training(training_control):
        if isinstance(training_control, TrainingControl):
            training_control.request_stop()
        # The terminal training update owns the label. A late Stop response must
        # not replace its restored label with "Stopping…" after training finishes.
        return gr.update(interactive=False)

    event = button.click(stream_and_reset, [*inputs, control_state], all_outputs,
                         concurrency_id=concurrency_id, show_progress="minimal")
    # It must bypass the serialized model-action queue so it can signal a run
    # currently holding that queue. No tensors or optimizer objects are changed.
    stop_button.click(stop_training, [control_state], [stop_button], queue=False)
    return event
