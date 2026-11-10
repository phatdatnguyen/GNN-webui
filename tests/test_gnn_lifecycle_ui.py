"""Resource ownership and full prediction exports through real Gradio sessions."""

import asyncio
from functools import wraps
import gc
import inspect
from pathlib import Path
import weakref

import gradio as gr
import pandas as pd
import pytest
import torch
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from run_utils import make_run_context, make_split_context, run_directory
from session_lifecycle import register_session_cleanup


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, 8),
    (bc3, bc3.binary_classification_3d_tab_content, 7),
    (rg, rg.regression_hybrid_tab_content, 8),
    (rg3, rg3.regression_3d_tab_content, 7),
]


class UI:
    def __init__(self, builder):
        with gr.Blocks() as self.blocks:
            builder()
        self.state = SessionState(self.blocks)
        self.values = {c._id: getattr(c, "value", None) for c in self.blocks.blocks.values()}

    def callback(self, name):
        return next(fn for fn in self.blocks.fns.values() if getattr(fn.fn, "__name__", "") == name)

    async def call(self, name, **arguments):
        fn = self.callback(name)
        parameter_names = list(inspect.signature(fn.fn).parameters)
        for name, value in arguments.items():
            component = fn.inputs[parameter_names.index(name)]
            if isinstance(component, gr.State):
                self.state[component._id] = value
            else:
                self.values[component._id] = value
        response = await asyncio.wait_for(self.blocks.process_api(
            fn, [None if isinstance(c, gr.State) else self.values[c._id] for c in fn.inputs],
            state=self.state, session_hash="lifecycle-test"), 20)
        assert len(response["data"]) == len(fn.outputs)
        for component, value in zip(fn.outputs, response["data"]):
            if isinstance(component, gr.State):
                continue
            self.values[component._id] = value
        return response


@pytest.mark.parametrize("module,builder,model_index", CASES, ids=[c[0].__name__ for c in CASES])
@pytest.mark.parametrize("success", [False, True])
def test_creation_releases_previous_optimizer_and_scheduler(monkeypatch, module, builder, model_index, success):
    original = module.on_create_model

    @wraps(original)
    def create(*args, **kwargs):
        message = "<span style='color: green;'>Created</span>" if success else "<span style='color: red;'>Failed</span>"
        return (message, *[gr.update() for _ in range(model_index - 1)],
                torch.nn.Linear(1, 1) if success else None, [], [], 0)

    monkeypatch.setattr(module, "on_create_model", create)
    ui = UI(builder)
    creation = ui.callback("on_create_model")
    optimization = ui.callback("on_create_optimizer")
    old = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(old.parameters())
    old(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    references = [weakref.ref(old), weakref.ref(old.weight), weakref.ref(old.weight.grad),
                  weakref.ref(optimizer.state[old.weight]["exp_avg"]), weakref.ref(optimizer), weakref.ref(scheduler)]
    ui.state[creation.outputs[model_index]._id] = old
    ui.state[optimization.outputs[1]._id] = optimizer
    ui.state[optimization.outputs[2]._id] = scheduler
    del old, optimizer, scheduler
    assert all(ref() is not None for ref in references)
    asyncio.run(ui.call("on_create_model"))
    gc.collect()
    assert all(ref() is None for ref in references)
    assert ui.state[optimization.outputs[1]._id] is None
    assert ui.state[optimization.outputs[2]._id] is None


@pytest.mark.parametrize("module,builder,model_index", CASES, ids=[c[0].__name__ for c in CASES])
def test_prediction_preview_exports_all_rows_and_releases_run(tmp_path, monkeypatch, module, builder, model_index):
    frame = pd.DataFrame({"SMILES": ["CCO"] * 457, "target (Prediction)": list(range(457))})
    original = module.on_predict

    @wraps(original)
    def predict(*args, **kwargs):
        return "<span style='color: green;'>Prediction completed.</span>", frame, gr.update(interactive=True)

    monkeypatch.setattr(module, "on_predict", predict)
    ui = UI(builder)
    split_frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCCC"], "target": [0, 1, 0, 1]})
    split = make_split_context(str(tmp_path), "dataset.csv", split_frame, [0, 1], [2, 3], 17, "Random")
    model = torch.nn.Linear(1, 1)
    model._run_context = make_run_context(split, "preview", 42, module.__name__, "Linear")
    run_path = Path(run_directory(model._run_context))
    prediction = ui.callback("on_predict")
    export = ui.callback("on_export_prediction")
    assert isinstance(export.inputs[2], gr.State)
    assert export.inputs[2] is prediction.outputs[-1]

    async def run():
        response = await ui.call("on_predict", model=model)
        assert len(response["data"][1]["data"]) == 200
        assert "457" in response["data"][0]
        assert ui.state[export.inputs[2]._id] is frame
        await ui.call("on_export_prediction", working_directory_path=str(tmp_path), prediction_file_name="all_rows")
        pd.testing.assert_frame_equal(pd.read_csv(run_path / "all_rows.csv"), frame)
        # Reprocessing prediction input must invalidate its previous results even if it fails.
        response = await ui.call("on_process_prediction_data", prediction_dataset_file=None)
        assert "color: red" in response["data"][0]
        assert ui.state[export.inputs[2]._id] is None
        await ui.call("on_predict", model=model)
        release = ui.callback("release_current_run")
        # Every resource reset is exercised through Gradio, including transformers/loaders/results.
        for component in release.outputs:
            if isinstance(component, gr.State):
                ui.state[component._id] = object()
        await ui.call("release_current_run")
        for component in release.outputs:
            if isinstance(component, gr.State):
                assert ui.state[component._id] in (None, [], 0)
        assert ui.state[export.inputs[0]._id] == str(tmp_path)
        assert (run_path / "all_rows.csv").exists()

    asyncio.run(run())


def test_disconnect_detaches_only_its_session_and_active_callback_finishes():
    class Resource:
        pass

    async def run():
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def in_flight(resource):
            entered.set()
            await finish.wait()
            return resource

        with gr.Blocks() as blocks:
            state = gr.State()
            button = gr.Button()
            button.click(in_flight, state, state)
            register_session_cleanup(blocks)
        active = blocks.state_holder["closing"]
        other = blocks.state_holder["other"]
        resource = Resource()
        reference = weakref.ref(resource)
        active[state._id] = resource
        other[state._id] = Resource()
        original_other = other[state._id]
        # A rendered payload belongs to the same session owner and must not survive it.
        active.config_values[button._id]["props"]["value"] = "large rendered result"
        callback = next(fn for fn in blocks.fns.values() if fn.fn.__name__ == "in_flight")
        unload = next(fn for fn in blocks.fns.values() if fn.fn.__name__ == "on_session_unload")
        pending = asyncio.create_task(blocks.process_api(callback, [None], state=active, session_hash="closing"))
        await asyncio.wait_for(entered.wait(), 20)
        del resource
        await asyncio.wait_for(blocks.process_api(unload, [], state=active, session_hash="closing",
                                 request=gr.Request(session_hash="closing")), 20)
        assert "closing" not in blocks.state_holder.session_data
        assert "closing" not in blocks.state_holder.time_last_used
        assert blocks.state_holder.session_data["other"] is other
        assert other[state._id] is original_other
        assert reference() is not None  # Ongoing work retains ownership; no mutation or forced disposal.
        finish.set()
        await asyncio.wait_for(pending, 20)
        assert active[state._id] is reference()
        assert "closing" not in blocks.state_holder.session_data  # Postprocessing did not reinsert it.
        del active, pending
        gc.collect()
        assert reference() is None

    asyncio.run(run())


def test_disconnect_callback_receives_session_in_mounted_fastapi():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    with gr.Blocks() as blocks:
        state = gr.State()
        register_session_cleanup(blocks)
    app = gr.mount_gradio_app(FastAPI(), blocks, path="/")
    unload = next(fn for fn in blocks.fns.values() if fn.fn.__name__ == "on_session_unload")
    blocks.state_holder["closing"][state._id] = object()
    blocks.state_holder["other"][state._id] = "preserved"
    with TestClient(app) as client:
        # Invoke the registered unload event through the same mounted process API
        # used by Gradio's heartbeat disconnect handler, including Request injection.
        response = client.post("/gradio_api/run/on_session_unload", json={
            "data": [], "fn_index": unload._id, "session_hash": "closing"})
    assert response.status_code == 200, response.text
    assert "closing" not in blocks.state_holder.session_data
    assert blocks.state_holder["other"][state._id] == "preserved"


def test_disconnect_removes_queued_work_without_canceling_other_sessions():
    from gradio.queueing import Event, EventQueue

    async def run():
        with gr.Blocks() as blocks:
            state = gr.State()
            button = gr.Button()
            button.click(lambda value: value, state, state)
            register_session_cleanup(blocks)
        fn = next(fn for fn in blocks.fns.values() if fn.fn.__name__ == "<lambda>")
        unload = next(fn for fn in blocks.fns.values() if fn.fn.__name__ == "on_session_unload")
        closing = blocks.state_holder["closing"]
        closing[state._id] = "old resource"
        queue = blocks._queue
        pending = Event("closing", fn, None, None)
        active = Event("closing", fn, None, None)
        other = Event("other", fn, None, None)
        event_queue = EventQueue(fn.concurrency_id, 1)
        event_queue.queue = [pending, other]
        queue.event_queue_per_concurrency_id[fn.concurrency_id] = event_queue
        queue.active_jobs = [[active]]
        queue.event_ids_to_events.update({event._id: event for event in [pending, active, other]})
        queue.pending_event_ids_session["closing"] = {pending._id, active._id}
        queue.pending_event_ids_session["other"] = {other._id}
        await asyncio.wait_for(blocks.process_api(unload, [], state=closing, session_hash="closing",
                                                 request=gr.Request(session_hash="closing")), 20)
        assert event_queue.queue == [other]
        assert pending._id not in queue.event_ids_to_events
        assert active.alive is False
        assert other.alive is True
        assert queue.pending_event_ids_session["other"] == {other._id}
        assert closing[state._id] == "old resource"
        assert "closing" not in blocks.state_holder.session_data

    asyncio.run(run())
