"""Run metadata and atomic UI resets shared by the four GNN workflows."""

from copy import deepcopy
from functools import wraps
import html
import inspect
import os
from numbers import Integral

import gradio as gr

from run_utils import ensure_run_directory, list_saved_models, make_run_context, read_run_context_for_file, rebase_run_context, run_directory
from run_utils import ensure_model_split as validate_model_split
from utils import load_model_checkpoint, prefixed_feature_directory, sanitize_path_component
from feature_provenance import capture_feature_sources as snapshot_feature_sources


def capture_feature_sources(dataset, graph_directory_path=None, fingerprint_directory_path=None):
    """Snapshot feature settings and loaded fingerprints with the processed split."""
    dataset.feature_sources = snapshot_feature_sources(
        graph_directory=graph_directory_path, fingerprint_directory=fingerprint_directory_path,
        fingerprint_df=getattr(dataset, "fingerprint_df", None))


def attach_model_run(model, dataset, model_name, model_seed, workflow, parameters):
    """Freeze named UI runs while preserving unnamed direct-handler callers."""
    if model_name is None:
        return
    split_context = getattr(dataset, "split_context", None)
    if split_context is None:
        raise ValueError("Please process the data before creating a named model.")
    settings = {key: value for key, value in parameters.items()
                if key not in {"dataset", "model_name", "variance_threshold", "pca"}
                and isinstance(value, (str, int, float, bool, type(None)))}
    if "datatype" in parameters:
        settings["datatype"] = str(parameters["datatype"])
    settings["architecture"] = repr(model)
    settings["architecture_modules"] = {name: repr(module) for name, module in model.named_modules()}
    settings["load_data"] = getattr(dataset, "load_data", "3D graphs")
    settings["target_names"] = getattr(dataset, "target_column_names", [getattr(dataset, "target_column_name", None)])
    settings["feature_sources"] = deepcopy(getattr(dataset, "feature_sources", {}))
    for transformer_name in ("pca", "variance_threshold"):
        if parameters.get(transformer_name) is not None:
            settings[transformer_name] = parameters[transformer_name].get_params(deep=False)
    context = make_run_context(split_context, model_name, model_seed, workflow,
                               type(model).__name__, parameters=settings)
    model._run_context = deepcopy(context)


def checkpoint_run_context(model, checkpoint_path, checkpoint_name, checkpoint, workdir):
    """Use validated portable metadata; legacy imports retain their named UI run."""
    context = read_run_context_for_file(checkpoint_path)
    saved_context = checkpoint.get("run_context")
    if saved_context is not None:
        saved_context = rebase_run_context(saved_context, workdir)
        if context is not None and context != saved_context:
            raise ValueError("Checkpoint run metadata does not match its run directory.")
        context = saved_context
    if context is not None:
        saved_parameters = context.get("parameters", {})
        if (saved_parameters.get("architecture", repr(model)) != repr(model)
                or saved_parameters.get("architecture_modules", {name: repr(module) for name, module in model.named_modules()})
                != {name: repr(module) for name, module in model.named_modules()}):
            raise ValueError("Checkpoint architecture settings differ from the current model. Re-create the original architecture first.")
        return deepcopy(context), " Recorded run metadata restored; training and evaluation require its original processed split."
    current = getattr(model, "_run_context", None)
    if current is None:
        return None, ""
    context = deepcopy(current)
    context.update(legacy_checkpoint=True, initialization_seed_verified=False,
                   source_checkpoint=checkpoint_name)
    return context, " Legacy checkpoint imported into this named run; its original initialization seed is unknown."


def load_run_checkpoint(model, checkpoint_path, checkpoint_name, workdir):
    prepared = {}

    def validate_metadata(checkpoint):
        context, note = checkpoint_run_context(model, checkpoint_path, checkpoint_name, checkpoint, workdir)
        prepared.update(context=context, note=note)

    checkpoint = load_model_checkpoint(model, checkpoint_path, validate_checkpoint=validate_metadata)
    model._run_context = prepared["context"]
    return checkpoint, prepared["note"]


def ensure_model_split(model, dataset):
    """Prevent training/evaluating a restored run against another processed split."""
    context = getattr(model, "_run_context", None)
    if context is None:
        return
    while dataset is not None and not hasattr(dataset, "split_context") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    current = getattr(dataset, "split_context", None)
    if current is None:
        raise ValueError("Please process the original dataset and data split used by this model before training or evaluation.")
    validate_model_split(model, dataset)
    recorded_features = context.get("parameters", {}).get("feature_sources")
    if recorded_features is not None and recorded_features != getattr(dataset, "feature_sources", None):
        raise ValueError("This model belongs to different graph or fingerprint features. Process its original features or create a new named model.")


def capture_split_outputs(callback, dataset_output_index, similarity_controls):
    """Append a snapshot and cleared similarity outputs to a processing response."""
    @wraps(callback)
    def process_with_context(*args, **kwargs):
        result = callback(*args, **kwargs)
        dataset = result[dataset_output_index]
        context = deepcopy(getattr(dataset, "split_context", None))
        return (*result, context, *similarity_controls.reset_updates().values())
    return process_with_context


def show_model_run(callback, model_output_index, restore_controls=False):
    """Display the captured run separately from controls for the next model."""
    @wraps(callback)
    def action_with_run(*args, **kwargs):
        result = callback(*args, **kwargs)
        if restore_controls and "color: green" not in result[0]:
            return (*result, gr.update(), gr.update(), gr.update())
        context = getattr(result[model_output_index], "_run_context", None)
        display = gr.update(value=run_directory(context) if context is not None else None)
        if restore_controls:
            name = gr.update(value=context["model_name"]) if context is not None else gr.update()
            seed = gr.update(value=context["model_seed"]) if context is not None else gr.update()
            return (*result, display, name, seed)
        return (*result, display)
    return action_with_run


def route_model_export(callback):
    """Route a UI export using the fitted model, never current textbox values."""
    @wraps(callback)
    def export_for_model(workdir, *args):
        try:
            model = args[-1]
            if model is None:
                raise ValueError("Please create or load a model first.")
            context = getattr(model, "_run_context", None)
            if context is None:
                raise ValueError("Please create a named model before exporting its results.")
            if not workdir or os.path.realpath(workdir) != os.path.realpath(context["working_directory"]):
                raise ValueError("Please reopen this model's working directory before exporting its results.")
            output_directory = ensure_run_directory(context)
            result = callback(output_directory, *args[:-1])
            return result[0], os.listdir(workdir) if workdir and os.path.isdir(workdir) else gr.update()
        except Exception as exc:
            return f"<span style='color: red;'>Error exporting results: {html.escape(str(exc))}</span>", gr.update()
    signature = inspect.signature(callback)
    export_for_model.__signature__ = signature.replace(parameters=[
        *signature.parameters.values(), inspect.Parameter("model", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)])
    return export_for_model


def loss_epoch_numbers(train_losses, val_losses, trained_epochs=None):
    """Identify completed epochs in histories that may begin after a checkpoint."""
    if train_losses is None or val_losses is None or len(train_losses) != len(val_losses):
        raise ValueError("Training and validation loss histories must have the same length.")
    count = len(train_losses)
    completed = count if trained_epochs is None else trained_epochs
    if isinstance(completed, bool) or not isinstance(completed, Integral) or completed < count:
        raise ValueError("Completed epoch count must be an integer at least as large as the loss history.")
    return range(completed - count + 1, completed + 1)


def bind_working_directory(component, event, callback, inputs, outputs, *, reset_updates,
                           concurrency_id=None, refresh_callback=None, refresh_inputs=(),
                           refresh_outputs=(), path_output=None):
    """Reset state and refresh selectors atomically, even when directory names match."""
    inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
    all_outputs = list(dict.fromkeys([*outputs, *reset_updates, *refresh_outputs]))
    signature = inspect.signature(callback)
    parameters = list(signature.parameters.values())
    if refresh_callback is not None:
        parameters.extend(list(inspect.signature(refresh_callback).parameters.values())[1:])
    wrapped_signature = signature.replace(parameters=parameters)

    @wraps(callback)
    def open_and_reset(*args, **kwargs):
        bound = wrapped_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = list(bound.arguments.values())
        result = dict(zip(outputs, callback(*values[:len(inputs)]), strict=True))
        result.update({component: deepcopy(value) for component, value in reset_updates.items()})
        if refresh_callback is not None:
            try:
                refreshed = refresh_callback(result[path_output], *values[len(inputs):])
                result.update(zip(refresh_outputs, refreshed, strict=True))
            except Exception as exc:
                # A refresh failure must still publish the new path and its state reset.
                gr.Warning(f"Could not refresh the working directory: {exc}")
                result.update((output, gr.update(choices=[], value=None) if isinstance(output, gr.Dropdown)
                               else gr.update(value=None)) for output in refresh_outputs)
        return result

    open_and_reset.__signature__ = wrapped_signature
    return getattr(component, event)(open_and_reset, [*inputs, *refresh_inputs], all_outputs,
                                     concurrency_id=concurrency_id)


def bind_checkpoint_save(button, callback, inputs, outputs, checkpoint_dropdown, *, concurrency_id=None):
    """Publish a saved checkpoint and its selection without a directory-state event."""
    signature = inspect.signature(callback)

    @wraps(callback)
    def save_and_refresh(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        response = callback(*args, **kwargs)
        result = dict(zip(outputs, response, strict=True))
        result[checkpoint_dropdown] = gr.update()
        if "color: green" in response[0]:
            try:
                values = bound.arguments
                workdir = values["working_directory_path"]
                context = getattr(values["model"], "_run_context", None)
                filename = f"{sanitize_path_component(values['checkpoint_name'], 'checkpoint name')}_{values['trained_epochs']}.ckpt"
                selected = f"{os.path.basename(run_directory(context))}/{filename}" if context else filename
                choices = list_saved_models(workdir, ".ckpt")
                result[checkpoint_dropdown] = gr.update(choices=choices, value=selected if selected in choices else None)
            except Exception as exc:
                result[outputs[0]] += (" <span style='color: orange;'>The checkpoint was saved, but its list "
                                       f"could not be refreshed: {html.escape(str(exc))}. Reopen the working directory to retry.</span>")
        return result

    return button.click(save_and_refresh, inputs, [*outputs, checkpoint_dropdown], concurrency_id=concurrency_id)


def reset_checkpoint_optimizer(callback):
    """Loaded weights need a fresh optimizer; failed loads preserve its state."""
    signature = inspect.signature(callback)
    wrapped_signature = signature.replace(parameters=[
        *signature.parameters.values(),
        inspect.Parameter("optimizer", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
        inspect.Parameter("scheduler", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
    ])

    @wraps(callback)
    def load_and_reset(*args, **kwargs):
        bound = wrapped_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        optimizer = bound.arguments.pop("optimizer")
        scheduler = bound.arguments.pop("scheduler")
        result = callback(**bound.arguments)
        if "color: green" in result[0]:
            message = result[0].replace("</span>", " Create a new optimizer before continuing training.</span>")
            return (message, *result[1:], None, None, gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=False))
        return (*result, optimizer, scheduler, gr.update(), gr.update(), gr.update())

    load_and_reset.__signature__ = wrapped_signature
    return load_and_reset


def reset_created_optimizer(callback):
    """Release the previous model's optimizer and scheduler on every creation attempt."""
    @wraps(callback)
    def create_and_reset(*args, **kwargs):
        result = callback(*args, **kwargs)
        return (*result, None, None, gr.update(value=None), gr.update(interactive=False))
    return create_and_reset


def cache_prediction_rows(callback, preview_rows=200):
    """Retain complete predictions server-side and send only a preview to the browser."""
    @wraps(callback)
    def predict_with_preview(*args, **kwargs):
        status, frame, export_button = callback(*args, **kwargs)
        if "color: green" not in status or frame is None:
            return status, gr.update(value=None), gr.update(interactive=False), None
        count = len(frame)
        message = (f" Showing {min(count, preview_rows):,} of {count:,} rows. "
                   "Export includes all rows.")
        status = status.replace("</span>", message + "</span>")
        return status, frame.head(preview_rows).copy(), export_button, frame
    return predict_with_preview


def bind_release_run(button, status_component, reset_updates, *, concurrency_id=None):
    """Drop all resources owned by this tab while preserving its directory and files."""
    def release_current_run():
        result = {component: deepcopy(value) for component, value in reset_updates.items()}
        result[status_component] = (
            "<span style='color: green;'>Current run released. "
            "Saved files are unchanged. Process data to start another run.</span>")
        return result

    return button.click(release_current_run, [], [status_component, *reset_updates],
                        concurrency_id=concurrency_id)


def build_loader_controls():
    """Conservative performance settings shared by training and prediction loading."""
    with gr.Group():
        gr.Markdown("**Data loading**")
        with gr.Row():
            graph_cache_mb = gr.Slider(
                minimum=0, maximum=2048, step=32, value=128,
                label="Graph cache limit (MiB)",
                info="Per processed dataset, including prediction data. Set 0 for disk loading.")
            loader_workers = gr.Dropdown(
                choices=[0, 2, 4], value=0, label="DataLoader workers",
                info="0 uses the main process. Extra workers may increase memory use.")
            pin_memory = gr.Checkbox(value=False, label="Pin memory for CUDA transfers")
    return graph_cache_mb, loader_workers, pin_memory


def directory_reset_updates(clear_states, loss_states, epoch_state, disable_buttons,
                            clear_components, similarity_controls):
    updates = dict.fromkeys(clear_states, None)
    updates.update({state: [] for state in loss_states})
    updates[epoch_state] = 0
    updates.update({button: gr.update(interactive=False) for button in disable_buttons})
    updates.update({component: gr.update(value=None) for component in clear_components})
    updates.update(similarity_controls.reset_updates())
    return updates


def bind_artifact_update(button, callback, inputs, outputs, *, dataset_state,
                         reset_updates, kind, concurrency_id=None):
    """Invalidate processed data when an upload/extraction touches its source files.

    Unrelated prediction uploads preserve the current model and similarity results.
    Failed uploads are atomic and preserve their input; extraction failures can follow
    partial publication, so a touched training source is conservatively invalidated.
    """
    if kind not in {"upload", "graphs", "fingerprints"}:
        raise ValueError("Artifact kind must be upload, graphs or fingerprints.")
    state_index = 5 if kind == "upload" else len(inputs)
    signature = inspect.signature(callback)
    parameters = list(signature.parameters.values())
    state_parameter = inspect.Parameter("processed_dataset", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        default=None)
    wrapped_signature = signature.replace(parameters=[*parameters[:state_index], state_parameter,
                                                        *parameters[state_index:]])
    all_inputs = [*inputs[:state_index], dataset_state, *inputs[state_index:]]
    all_outputs = list(dict.fromkeys([*outputs, *reset_updates]))

    @wraps(callback)
    def update_and_invalidate(*args, **kwargs):
        bound = wrapped_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        dataset = bound.arguments.pop("processed_dataset")
        values = [bound.arguments[parameter.name] for parameter in parameters[:len(inputs)]]
        context = getattr(dataset, "split_context", None)
        touched = False
        if context:
            path = values[1] if kind == "upload" else values[0]
            same_directory = (path is not None and
                              os.path.realpath(path) == os.path.realpath(context["working_directory"]))
            if same_directory:
                if kind == "upload":
                    touched = f"{str(values[2]).strip()}.csv" == context["dataset_file"]
                else:
                    feature_key = "graph_directory" if kind == "graphs" else "fingerprint_directory"
                    source = getattr(dataset, "feature_sources", {}).get(feature_key)
                    try:
                        destination = prefixed_feature_directory(
                            values[2], "graph" if kind == "graphs" else "fingerprint")
                    except ValueError:
                        destination = None
                    touched = (values[1] == context["dataset_file"] or
                               (source is not None and destination == source))
        try:
            result = callback(**bound.arguments)
            original_updates = dict(zip(outputs, result, strict=True))
            succeeded = isinstance(result[0], str) and "color: green" in result[0]
        except Exception as exc:
            original_updates = {outputs[0]: f"<span style='color: red;'>Error updating data: {html.escape(str(exc))}</span>"}
            succeeded = False
        updates = ({component: deepcopy(value) for component, value in reset_updates.items()}
                   if touched and (kind != "upload" or succeeded) else {})
        # Preserve the successful upload preview and the current directory listing.
        updates.update(original_updates)
        return updates

    update_and_invalidate.__signature__ = wrapped_signature
    return button.click(update_and_invalidate, all_inputs, all_outputs, concurrency_id=concurrency_id)
