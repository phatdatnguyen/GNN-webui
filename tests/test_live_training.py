"""Live GNN loss updates retain coherent state when stopped, resumed, or failed."""

import asyncio
import inspect
from types import SimpleNamespace

import gradio as gr
import pytest
import torch
from gradio.state_holder import SessionState
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from conftest import FakeProgress
from gnn_runtime import make_loader
from training_control import InterruptibleLoader, TrainingControl, TrainingStopped


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content),
    (bc3, bc3.binary_classification_3d_tab_content),
    (rg, rg.regression_hybrid_tab_content),
    (rg3, rg3.regression_3d_tab_content),
]


class TrainingUI:
    """Advance real Gradio generator responses without starting an HTTP server."""

    def __init__(self, builder):
        with gr.Blocks() as self.blocks:
            builder()
        self.session = SessionState(self.blocks)
        self.values = {c._id: getattr(c, "value", None) for c in self.blocks.blocks.values()}
        self.properties = {}
        self.training = self.callback("on_train")
        self.stop = self.callback("stop_training")
        self.train_button = next(c for c in self.blocks.blocks.values()
                                 if isinstance(c, gr.Button) and c.value == "Train")
        self.stop_button = next(c for c in self.blocks.blocks.values()
                                if isinstance(c, gr.Button) and c.value == "Stop training")

    def callback(self, name):
        return next(fn for fn in self.blocks.fns.values() if getattr(fn.fn, "__name__", "") == name)

    def set_arguments(self, **arguments):
        names = list(inspect.signature(self.training.fn).parameters)
        for name, value in arguments.items():
            component = self.training.inputs[names.index(name)]
            if isinstance(component, gr.State):
                self.session[component._id] = value
            else:
                self.values[component._id] = value

    async def call(self, callback=None, *, iterator=None, session=None):
        callback = callback or self.training
        response = await asyncio.wait_for(self.blocks.process_api(
            callback, [None if isinstance(c, gr.State) else self.values[c._id]
                       for c in callback.inputs],
            state=session or self.session, iterator=iterator,
            session_hash="other" if session is not None else "live-training"), 20)
        assert len(response["data"]) == len(callback.outputs)
        for component, value in zip(callback.outputs, response["data"], strict=True):
            if isinstance(component, gr.State):
                continue
            if isinstance(value, dict) and value.get("__type__") == "update":
                self.properties.setdefault(component._id, {}).update(value)
                if "value" in value:
                    self.values[component._id] = value["value"]
            else:
                self.values[component._id] = value
        return response

    def state(self, output_index):
        return self.session[self.training.outputs[output_index]._id]

    def interactive(self, button):
        return self.properties[button._id]["interactive"]


def training_values(model=None):
    model = model if model is not None else torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return dict(model=model, dataset=SimpleNamespace(load_data="Molecular fingerprints"),
                train_dataloader=object(), val_dataloader=object(),
                device="cpu", datatype=torch.float32, output_scaler=MinMaxScaler().fit([[0.0], [1.0]]),
                optimizer=optimizer, scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                epochs=4, trained_epochs=10, train_losses=[], val_losses=[],
                gnn_model_tab="GCN", customgcn_convolutional_layer_name="GCNConv",
                variance_threshold=None, pca=None)


@pytest.mark.parametrize("module,builder", CASES, ids=[m.__name__ for m, _ in CASES])
def test_ui_streams_epochs_stops_and_resumes_only_its_own_session(module, builder, monkeypatch):
    train_calls = []

    def train(*args):
        train_calls.append(True)
        return 1.0 / len(train_calls)

    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(module, "validation", lambda *args: 0.6)
    ui = TrainingUI(builder)
    values = training_values()
    ui.set_arguments(**{name: value for name, value in values.items()
                        if name in inspect.signature(ui.training.fn).parameters})
    assert inspect.isgeneratorfunction(ui.training.fn)
    assert ui.stop.queue is False
    assert ui.stop_button.interactive is False
    assert ui.training.concurrency_id == ui.callback("on_create_optimizer").concurrency_id
    evaluation_state = ui.callback("on_evaluate").outputs[1]
    ui.session[evaluation_state._id] = "obsolete evaluated rows"
    control_component = ui.stop.inputs[0]
    second_session = SessionState(ui.blocks)

    async def run():
        initial = await ui.call()
        assert initial["is_generating"] and ui.state(3) == 10
        assert train_calls == []  # The browser gets a response before the first epoch.
        assert ui.session[evaluation_state._id] is None
        assert ui.interactive(ui.train_button) is False
        assert ui.interactive(ui.stop_button) is True
        control = ui.session[control_component._id]
        other_control = second_session[control_component._id]
        assert control is not other_control and control.is_running

        # A Stop request from another browser cannot interrupt this stream.
        await ui.call(ui.stop, session=second_session)
        assert not control.stop_requested
        first = await ui.call(iterator=initial["iterator"])
        assert ui.state(3) == 11 and len(ui.state(4)) == len(ui.state(5)) == 1
        first_history = ui.state(4)
        first_plot = first["data"][1]
        second = await ui.call(iterator=first["iterator"])
        assert ui.state(3) == 12 and len(ui.state(4)) == len(ui.state(5)) == 2
        assert first_history == [1.0]  # Old streamed snapshots must not grow in place.
        assert first_plot != second["data"][1]
        assert values["scheduler"].last_epoch == 2

        await ui.call(ui.stop)
        stopped = await ui.call(iterator=second["iterator"])
        assert "stopped" in stopped["data"][0].lower()
        assert ui.state(3) == 12 and len(train_calls) == 2
        assert values["scheduler"].last_epoch == 2
        assert not control.is_running
        assert ui.interactive(ui.train_button) is True
        assert ui.interactive(ui.stop_button) is False
        finished = await ui.call(iterator=stopped["iterator"])
        assert not finished["is_generating"]

        # Resume keeps the current optimizer and continues cumulative epoch numbers.
        ui.set_arguments(epochs=1)
        restarted = await ui.call()
        assert control.is_running and not control.stop_requested
        completed = await ui.call(iterator=restarted["iterator"])
        assert "completed" in completed["data"][0].lower()
        assert ui.state(3) == 13 and len(ui.state(4)) == len(ui.state(5)) == 3
        assert values["scheduler"].last_epoch == 3
        assert not control.is_running
        assert ui.interactive(ui.train_button) is True and ui.interactive(ui.stop_button) is False
        assert not (await ui.call(iterator=completed["iterator"]))["is_generating"]

    asyncio.run(run())


@pytest.mark.parametrize("module,builder", CASES, ids=[m.__name__ for m, _ in CASES])
def test_stream_errors_publish_completed_history_and_reset_buttons(module, builder, monkeypatch):
    calls = []

    def validate(*args):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("validation interrupted by test")
        return 0.6

    monkeypatch.setattr(module, "train", lambda *args: 0.5)
    monkeypatch.setattr(module, "validation", validate)
    ui = TrainingUI(builder)
    values = training_values()
    ui.set_arguments(**{name: value for name, value in values.items()
                        if name in inspect.signature(ui.training.fn).parameters})

    async def run():
        initial = await ui.call()
        first = await ui.call(iterator=initial["iterator"])
        failed = await ui.call(iterator=first["iterator"])
        assert "color: red" in failed["data"][0]
        assert "validation interrupted by test" in failed["data"][0]
        assert ui.state(3) == 11 and ui.state(4) == [0.5] and ui.state(5) == [0.6]
        assert values["scheduler"].last_epoch == 1
        assert ui.interactive(ui.stop_button) is False
        assert not ui.session[ui.stop.inputs[0]._id].is_running
        assert not (await ui.call(iterator=failed["iterator"]))["is_generating"]

        # Validation failures before training starts are returned immediately.
        ui.set_arguments(model=None)
        rejected = await ui.call()
        assert "color: red" in rejected["data"][0]
        assert len(calls) == 2 and ui.state(3) == 11
        assert ui.interactive(ui.stop_button) is False
        assert not ui.session[ui.stop.inputs[0]._id].is_running
        assert not (await ui.call(iterator=rejected["iterator"]))["is_generating"]

    asyncio.run(run())


class Tiny3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, z, pos, batch):
        return self.weight.expand(int(batch.max()) + 1, 1)


@pytest.mark.parametrize("module", [bc, rg, bc3, rg3], ids=lambda module: module.__name__)
@pytest.mark.parametrize("phase", ["training", "validation"])
def test_stop_between_real_batches_keeps_weights_but_omits_incomplete_epoch(module, phase):
    samples = [Data(fp=torch.ones(1, 2), y=torch.ones(1, 1), z=torch.tensor([6]),
                    pos=torch.zeros(1, 3), num_nodes=1) for _ in range(4)]
    loader = DataLoader(samples, batch_size=1)
    control = TrainingControl()
    control.start()
    model = Tiny3DModel() if module in (bc3, rg3) else torch.nn.Linear(2, 1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    values = training_values(model)
    values.update(train_dataloader=InterruptibleLoader(loader, control),
                  val_dataloader=InterruptibleLoader(loader, control), progress=FakeProgress())
    steps = []
    validation_batches = []

    def request_stop_after_update(optimizer, args, kwargs):
        steps.append(True)
        if phase == "training":
            control.request_stop()

    def request_stop_during_validation(model, inputs, output):
        if not model.training:
            validation_batches.append(True)
            if phase == "validation":
                control.request_stop()

    handle = values["optimizer"].register_step_post_hook(request_stop_after_update)
    validation_handle = model.register_forward_hook(request_stop_during_validation)
    try:
        updates = list(module.on_train.stream(**{
            name: values[name] for name in inspect.signature(module.on_train).parameters}))
    finally:
        handle.remove()
        validation_handle.remove()
        control.finish()
    stopped = updates[-1]
    assert stopped.finished and "stopped" in stopped[0].lower()
    assert "partial" in stopped[0].lower() or "incomplete" in stopped[0].lower()
    assert len(steps) == (1 if phase == "training" else len(loader))
    assert len(validation_batches) == (0 if phase == "training" else 1)
    assert stopped[3:] == (10, [], [])
    assert values["scheduler"].last_epoch == 0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_duplicate_start_and_generator_close_do_not_leave_control_owned(monkeypatch):
    monkeypatch.setattr(bc, "train", lambda *args: 0.5)
    monkeypatch.setattr(bc, "validation", lambda *args: 0.6)
    ui = TrainingUI(bc.binary_classification_hybrid_tab_content)
    values = training_values()
    ui.set_arguments(**{name: value for name, value in values.items()
                        if name in inspect.signature(ui.training.fn).parameters})

    async def run():
        initial = await ui.call()
        control = ui.session[ui.stop.inputs[0]._id]
        duplicate = await ui.call()
        assert "already running" in duplicate["data"][0]
        assert control.is_running and not control.stop_requested
        assert ui.interactive(ui.train_button) is False and ui.interactive(ui.stop_button) is True
        assert not (await ui.call(iterator=duplicate["iterator"]))["is_generating"]
        assert control.is_running

        # GeneratorExit (for example a closed stream) must release only its own run.
        await initial["iterator"].aclose()
        assert not control.is_running
        resumed = await ui.call()
        assert control.is_running and not control.stop_requested
        await ui.call(ui.stop)
        stopped = await ui.call(iterator=resumed["iterator"])
        assert "stopped" in stopped["data"][0].lower()
        assert ui.state(3) == 10 and ui.state(4) == ui.state(5) == []
        assert not control.is_running
        assert not (await ui.call(iterator=stopped["iterator"]))["is_generating"]

    asyncio.run(run())


def test_stopping_shuts_down_the_current_spawned_loader_workers():
    samples = [Data(x=torch.ones(1, 1), y=torch.ones(1, 1)) for _ in range(8)]
    source = make_loader(samples, 1, loader_workers=2)
    workers = []

    class TrackedLoader:
        def __iter__(self):
            iterator = iter(source)
            workers.extend(iterator._workers)
            return iterator

        def __getattr__(self, name):
            return getattr(source, name)

    control = TrainingControl()
    control.start()
    iterator = iter(InterruptibleLoader(TrackedLoader(), control))
    try:
        next(iterator)
        assert len(workers) == 2 and all(worker.is_alive() for worker in workers)
        control.request_stop()
        with pytest.raises(TrainingStopped):
            next(iterator)
        for worker in workers:
            worker.join(timeout=10)
        assert all(not worker.is_alive() for worker in workers)
    finally:
        iterator.close()
        control.finish()
