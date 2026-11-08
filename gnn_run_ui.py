"""Run metadata and atomic UI resets shared by the four GNN workflows."""

from copy import deepcopy
from functools import wraps
import html
import inspect
import os

import gradio as gr

from run_utils import ensure_run_directory, make_run_context, read_run_context_for_file, rebase_run_context, run_directory
from run_utils import ensure_model_split as validate_model_split
from utils import load_model_checkpoint, prefixed_feature_directory
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
        *signature.parameters.values(), inspect.Parameter("model", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return export_for_model


def bind_working_directory(component, event, callback, inputs, outputs, *, reset_updates,
                           concurrency_id=None):
    """Clear split and model state in the response that changes the workdir."""
    all_outputs = list(dict.fromkeys([*outputs, *reset_updates]))

    @wraps(callback)
    def open_and_reset(*args, **kwargs):
        result = dict(zip(outputs, callback(*args, **kwargs), strict=True))
        result.update({component: deepcopy(value) for component, value in reset_updates.items()})
        return result

    return getattr(component, event)(open_and_reset, inputs, all_outputs, concurrency_id=concurrency_id)


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
