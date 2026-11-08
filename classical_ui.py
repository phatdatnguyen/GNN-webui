"""Shared fingerprint-only workflows for fitted scikit-learn models."""

import html
import json
import os
import time

import gradio as gr
import pandas as pd

import classical_ml as ml
from classical_plots import make_evaluation_plots
from similarity_ui import build_similarity_controls, bind_similarity_controls
from ui_utils import dataset_cleanup_controls
import run_utils
import utils


def status(message, success=False):
    color = "green" if success else "red"
    return f"<span style='color: {color};'>{html.escape(str(message))}</span>"


def succeeded(message):
    return "color: green" in message


def utility_status(message):
    """Escape the contents of the shared extractor's HTML status envelope."""
    inner = message.split(">", 1)[1].rsplit("</span>", 1)[0] if message.startswith("<span") else message
    return status(inner, succeeded(message))


def model_summary(model):
    context = getattr(model, "run_context", None)
    run = (f"Model name: {context['model_name']}\nData split random seed: {context['split_seed']}\n"
           if context is not None else "")
    return (f"{run}{model.model_name}\nTargets ({len(model.target_names)}): {', '.join(model.target_names)}\n"
            f"Parameters: {json.dumps(model.parameters, sort_keys=True)}\n"
            f"Model random seed: {model.random_seed}")


class ClassicalWorkflow:
    """Stateless handlers; fitted models and datasets live in browser session state."""

    def __init__(self, task, data_root):
        self.task = task
        self.data_root = data_root
        self.directory_name = "Classification" if task == "classification" else "Regression"

    def get_working_directories(self):
        root = os.path.join(self.data_root(), self.directory_name)
        os.makedirs(root, exist_ok=True)
        return sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))

    def on_open_working_directory(self, name):
        try:
            name = utils.sanitize_path_component(name, "working directory name")
            path = os.path.join(self.data_root(), self.directory_name, name)
            os.makedirs(path, exist_ok=True)
            return status(f"Working directory opened: {name}.", True), path
        except Exception as exc:
            return status(f"Error opening working directory: {exc}"), None

    def on_upload_dataset(self, file_path, workdir, name, smiles_column, targets,
                          canonicalize_smiles=True, remove_duplicates=True, remove_salt=True):
        try:
            if not workdir or not os.path.isdir(workdir):
                raise ValueError("Please open a working directory first.")
            if not file_path:
                raise ValueError("Please select a dataset CSV file.")
            result, frame = utils.process_dataset_file(
                file_path, workdir, name, smiles_column, targets or "",
                canonicalize_smiles=canonicalize_smiles, remove_duplicates=remove_duplicates, remove_salt=remove_salt)
            return result, frame.head(10) if frame is not None else gr.update()
        except Exception as exc:
            return status(f"Error uploading dataset: {exc}"), gr.update()

    def on_extract_fingerprints(self, workdir, dataset_name, fp_dir, fingerprint_name,
                                radius, number_of_bits, progress=gr.Progress()):
        try:
            if not workdir:
                raise ValueError("Please open a working directory first.")
            return utility_status(utils.extract_molecule_fingerprints(
                workdir, dataset_name, fp_dir, fingerprint_name, radius, number_of_bits, progress))
        except Exception as exc:
            return status(f"Error extracting molecular fingerprints: {exc}")

    def on_process_data(self, workdir, dataset_name, fp_dir, test_ratio=0.2,
                        random_seed=0, split_method="Random"):
        try:
            processed = ml.process_data(workdir, dataset_name, fp_dir, self.task,
                                        test_ratio, random_seed, split_method)
            processed.split_context = run_utils.make_split_context(
                workdir, dataset_name, processed.data.df, processed.train_indices,
                processed.test_indices, processed.random_seed, processed.split_method,
                validation_indices=processed.val_indices)
            message = (f"{split_method} split: {len(processed.train_indices)} training, "
                       f"{len(processed.test_indices)} test molecules. Fit a model for this split.")
            if split_method == "Scaffold":
                message += " Whole scaffold groups can make the split sizes differ from the requested ratios."
            return status(message, True), processed
        except Exception as exc:
            return status(f"Error processing data: {exc}"), None

    def _make_model_context(self, fitted, processed, run_name):
        if processed is None or fitted.training_identity != processed.identity:
            raise ValueError("Process the model's original dataset and split before assigning its model run.")
        if getattr(processed, "split_context", None) is None:
            raise ValueError("Please process the data again to capture its split information.")
        return run_utils.make_run_context(
            processed.split_context, f"{self.task}_classical" if run_name is None else run_name, fitted.random_seed,
            workflow=f"{self.task}_classical", model_class=fitted.model_name,
            parameters={"estimator_parameters": fitted.parameters,
                        "fingerprint_config": fitted.fingerprint_config,
                        "training_identity": fitted.training_identity})

    def _model_run_directory(self, workdir, model):
        context = getattr(model, "run_context", None)
        if context is None:
            raise ValueError("This model has no captured run. Process its original data and split, then load it again before saving or exporting.")
        if not workdir or os.path.realpath(workdir) != os.path.realpath(context["working_directory"]):
            raise ValueError("Open this model's working directory before saving or exporting its results.")
        return run_utils.ensure_run_directory(context)

    def on_fit(self, processed, model_name, parameters, random_seed, current_model=None, run_name=None):
        try:
            if processed is None:
                raise ValueError("Please process the training data first.")
            run_name = utils.sanitize_path_component(
                f"{self.task}_classical" if run_name is None else run_name, "model name")
            fitted = ml.fit_model(processed, model_name, parameters, random_seed)
            fitted.run_context = self._make_model_context(fitted, processed, run_name)
            message = f"{fitted.model_name} fitted on {len(processed.train_indices)} training molecules."
            return status(message, True), fitted, model_summary(fitted)
        except Exception as exc:
            return status(f"Error fitting model: {exc}"), current_model, gr.update()

    def on_evaluate(self, model, processed, current_metrics=None, current_rows=None, target=None):
        try:
            if model is None:
                raise ValueError("Please fit or load a model first.")
            if processed is None:
                raise ValueError("Please process the model's original dataset and split first.")
            rows = ml.evaluate_predictions(model, processed, split="test")
            metrics = ml.metrics_from_predictions(self.task, rows, model.target_names)
            selected = target if target in model.target_names else model.target_names[0]
            first_plot, second_plot = make_evaluation_plots(self.task, rows, selected)
            return status(f"Test evaluation completed for {model.model_name}.", True), metrics, rows, selected, first_plot, second_plot
        except Exception as exc:
            return status(f"Error evaluating model: {exc}"), current_metrics, current_rows, gr.update(), gr.update(), gr.update()

    def on_predict(self, model, workdir, dataset_name, fp_dir, current_predictions=None):
        try:
            if model is None:
                raise ValueError("Please fit or load a model first.")
            data = ml.load_fingerprint_data(workdir, dataset_name, fp_dir, self.task, require_targets=False)
            predictions = ml.predict(model, data)
            return status(f"Predictions completed for {model.model_name}.", True), predictions
        except Exception as exc:
            return status(f"Error making predictions: {exc}"), current_predictions

    def on_save_model(self, workdir, name, model):
        try:
            if not workdir or model is None:
                raise ValueError("Please open a working directory and fit or load a model first.")
            name = utils.sanitize_path_component(name, "model file name")
            filename = name if name.endswith(".joblib") else f"{name}.joblib"
            directory = self._model_run_directory(workdir, model)
            ml.save_model(model, os.path.join(directory, filename))
            relative_name = os.path.relpath(os.path.join(directory, filename), workdir)
            return status(f"Fitted model saved: {relative_name}.", True)
        except Exception as exc:
            return status(f"Error saving model: {exc}")

    def on_load_model(self, workdir, filename, current_model=None, processed=None, run_name=None):
        try:
            if not workdir:
                raise ValueError("Please open a working directory first.")
            path = run_utils.resolve_saved_model(workdir, filename, ".joblib")
            fitted = ml.load_model(path, self.task)
            embedded = getattr(fitted, "run_context", None)
            context = run_utils.rebase_run_context(embedded, workdir) if embedded is not None else None
            manifest = run_utils.read_run_context_for_file(path)
            if context is not None and manifest is not None and context != manifest:
                raise ValueError("The model checkpoint and run manifest contain different run information.")
            context = context if context is not None else manifest
            if context is not None:
                if (context["workflow"] != f"{self.task}_classical"
                        or context["model_class"] != fitted.model_name
                        or context["model_seed"] != fitted.random_seed
                        or context["parameters"].get("training_identity", fitted.training_identity) != fitted.training_identity
                        or context["parameters"].get("estimator_parameters", fitted.parameters) != fitted.parameters
                        or context["parameters"].get("fingerprint_config", fitted.fingerprint_config) != fitted.fingerprint_config):
                    raise ValueError("The captured run does not match the loaded model's data, estimator settings or random seed.")
            elif (processed is not None and fitted.training_identity == processed.identity
                  and getattr(processed, "split_context", None) is not None):
                context = self._make_model_context(fitted, processed, run_name)
            fitted.run_context = context
            message = (f"Loaded {fitted.model_name}. Predictions are available; evaluation requires "
                       "the original dataset and split used to fit this model.")
            if context is None:
                message += " Before saving or exporting, process that original split and load this legacy model again to assign a model name."
            return status(message, True), fitted, model_summary(fitted)
        except Exception as exc:
            return status(f"Error loading model: {exc}"), current_model, gr.update()

    def on_export(self, workdir, name, frame, model=None):
        try:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError("Please produce results before exporting.")
            utils.sanitize_path_component(name, "export file name")
            directory = self._model_run_directory(workdir, model)
            filename = utils.export_result_csv(directory, name, frame)
            relative_name = os.path.relpath(os.path.join(directory, filename), workdir)
            return status(f"Results exported: {relative_name}.", True)
        except Exception as exc:
            return status(f"Error exporting results: {exc}")


def build_classical_tab(workflow, title):
    model_names = list(ml.MODEL_NAMES[workflow.task])
    parameter_specs = ml.MODEL_PARAMETERS[workflow.task]
    descriptions = getattr(ml, "MODEL_DESCRIPTIONS", getattr(ml, "MODEL_NOTES", {})).get(workflow.task, {})
    with gr.Tab(title) as tab:
        path_state = gr.State()
        processed_state = gr.State()
        split_context_state = gr.State()
        training_source_state = gr.State()
        model_state = gr.State()
        test_state = gr.State()
        evaluation_rows_state = gr.State()
        prediction_state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                directory = gr.Dropdown(label="Working Directory", choices=workflow.get_working_directories(),
                                        value="wd", allow_custom_value=True,
                                        info=f"Folders in Data/{workflow.directory_name}. Select one or enter a new name.")
                open_button = gr.Button("Create/Open Working Directory")
                refresh_button = gr.Button("Refresh contents")
                contents = gr.Dataframe(label="Contents in Working Directory", headers=["Name", "Type", "Modified"], interactive=False)
            with gr.Column(scale=3):
                status_html = gr.HTML()
                with gr.Accordion("Upload dataset"):
                    file_input = gr.File(label="Dataset CSV", file_types=[".csv"], type="filepath")
                    with gr.Row():
                        dataset_name = gr.Textbox(label="Dataset name", value="dataset",
                                                  info="Use a new name when uploading prediction data.")
                        smiles_column = gr.Textbox(label="SMILES column", value="SMILES")
                        targets = gr.Textbox(label="Target columns (separated by commas)", value="target",
                                             info="Leave blank when uploading an unlabeled prediction dataset.")
                    canonicalize_smiles_checkbox, remove_duplicates_checkbox, remove_salt_checkbox = dataset_cleanup_controls()
                    upload_button = gr.Button("Load dataset", interactive=False)
                    preview = gr.Dataframe(label="Dataset preview", interactive=False)
                with gr.Accordion("Molecular fingerprint extraction"):
                    extraction_dataset = gr.Dropdown(label="Dataset file", choices=[])
                    with gr.Row():
                        fingerprint_method = gr.Dropdown(label="Molecular fingerprint", choices=utils.FINGERPRINT_NAMES,
                                                         value="Morgan fingerprint")
                        fingerprint_name = gr.Textbox(label="Fingerprint directory", value="FP_fingerprints",
                                                      info="FP_ is added automatically when omitted. Use a separate folder for each dataset.")
                    with gr.Row():
                        radius = gr.Slider(label="Radius", minimum=1, maximum=5, value=2, step=1)
                        bits = gr.Slider(label="Number of bits", minimum=128, maximum=4096, value=2048, step=128)
                    extract_button = gr.Button("Extract molecular fingerprints", interactive=False)
                with gr.Accordion("Load and process data"):
                    with gr.Row():
                        training_dataset = gr.Dropdown(label="Training dataset file", choices=[])
                        training_fps = gr.Dropdown(label="Training fingerprint directory", choices=[], info="Extracted fingerprints in FP_ folders.")
                    with gr.Group():
                        with gr.Row():
                            split_method = gr.Radio(label="Split method", choices=["Random", "Scaffold"], value="Random")
                            test_ratio = gr.Slider(label="Test size", minimum=0, maximum=1, value=0.2, step=0.01)
                            split_seed = gr.Slider(label="Data split random seed", minimum=0, maximum=1000, value=0, step=1)
                    process_button = gr.Button("Process data", interactive=False)
                    similarity_controls = build_similarity_controls()
                with gr.Accordion("Classical ML model"):
                    parameter_groups = {}
                    parameter_components = []
                    parameter_keys = []
                    with gr.Row():
                        with gr.Column(scale=1):
                            model_choice = gr.Dropdown(label="Model", choices=model_names, value=model_names[0])
                        with gr.Column(scale=2):
                            for name in model_names:
                                with gr.Group(visible=name == model_names[0]) as group:
                                    description = descriptions.get(name)
                                    if description:
                                        gr.Markdown(description)
                                    for key, spec in parameter_specs[name].items():
                                        label = spec.get("label", key.replace("_", " ").capitalize())
                                        if spec["type"] == "choice":
                                            component = gr.Dropdown(label=label, choices=spec["choices"], value=spec["default"])
                                        else:
                                            component = gr.Number(label=label, value=spec["default"],
                                                                  minimum=spec.get("min"), maximum=spec.get("max"),
                                                                  precision=0 if spec["type"] == "int" else None,
                                                                  info="0 means unlimited." if key == "max_depth" and spec.get("min") == 0 else None)
                                        parameter_components.append(component)
                                        parameter_keys.append((name, key))
                                parameter_groups[name] = group
                    gr.Markdown("Settings apply to the next fit. Models fit the training split; model evaluation uses the held-out test split.")
                    with gr.Row():
                        run_name = gr.Textbox(label="Model name", value=f"{workflow.task}_classical",
                                              info="Next fitted run is saved in MODEL_<model name>_<model seed>.")
                        model_seed = gr.Slider(label="Model random seed", minimum=0, maximum=1000, value=0, step=1)
                    fit_button = gr.Button("Fit model", interactive=False)
                    fitted_summary = gr.Textbox(label="Fitted model", lines=5, interactive=False)
                    with gr.Row():
                        save_name = gr.Textbox(label="Model file name", value="classical_model")
                        save_button = gr.Button("Save fitted model", interactive=False)
                        saved_model = gr.Dropdown(label="Saved model", choices=[], info="Models inside MODEL_ folders.")
                        load_button = gr.Button("Load fitted model", interactive=False)
                    gr.Markdown("Only load saved model files from sources you trust.")
                with gr.Accordion("Model evaluation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            evaluate_button = gr.Button("Evaluate test split", interactive=False)
                            output_target = gr.Dropdown(label="Output", choices=[], interactive=False)
                            test_table = gr.Dataframe(label="Evaluation metrics", interactive=False)
                            test_name = gr.Textbox(label="Metrics export file", value="test_metrics")
                            export_test = gr.Button("Export test metrics (all targets)", interactive=False)
                            evaluation_name = gr.Textbox(label="Evaluation export file", value="evaluation")
                            export_evaluation = gr.Button("Export evaluation rows (all targets)", interactive=False)
                        with gr.Column(scale=2):
                            if workflow.task == "classification":
                                first_plot = gr.Plot(label="Confusion matrix")
                                second_plot = gr.Plot(label="ROC curve")
                                evaluation_plots = [first_plot, second_plot]
                            else:
                                first_plot = gr.Plot(label="Actual vs. predicted")
                                evaluation_plots = [first_plot]
                with gr.Accordion("Make predictions"):
                    gr.Markdown("Training and prediction fingerprints must use the same method and settings. Legacy folders do not record these settings; re-extract them to enable compatibility checks.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            prediction_dataset = gr.Dropdown(label="Prediction dataset file", choices=[])
                            prediction_fps = gr.Dropdown(label="Prediction fingerprint directory", choices=[], info="Extracted fingerprints in FP_ folders.")
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    predict_button = gr.Button("Predict", interactive=False)
                                with gr.Column(scale=1):
                                    prediction_name = gr.Textbox(label="Prediction export file", value="predictions")
                                    export_prediction = gr.Button("Export predictions", interactive=False)
                            prediction_table = gr.Dataframe(label="Predictions", interactive=False)

    selection_components = [extraction_dataset, training_dataset, training_fps,
                            prediction_dataset, prediction_fps, saved_model]
    result_components = [test_table, prediction_table, *evaluation_plots]
    result_states = [test_state, evaluation_rows_state, prediction_state]
    export_buttons = [export_test, export_evaluation, export_prediction]
    dependent_buttons = [fit_button, evaluate_button, predict_button, save_button, *export_buttons]
    concurrency_id = f"classical-{workflow.task}-{model_state._id}"

    def reset(clear_processed=True):
        updates = {model_state: None, fitted_summary: gr.update(value=None),
                   output_target: gr.update(choices=[], value=None, interactive=False)}
        updates.update({state: None for state in result_states})
        updates.update({component: gr.update(value=None) for component in result_components})
        updates.update({button: gr.update(interactive=False) for button in dependent_buttons})
        if clear_processed:
            updates[processed_state] = None
            updates[split_context_state] = None
            updates[training_source_state] = None
            updates.update(similarity_controls.reset_updates())
        return updates

    def refresh_directory(path, *selected):
        try:
            names = os.listdir(path) if path and os.path.isdir(path) else []
            files = sorted(name for name in names if os.path.isfile(os.path.join(path, name)))
            csvs = [name for name in files if name.endswith(".csv")]
            models = run_utils.list_saved_models(path, ".joblib") if path and os.path.isdir(path) else []
            fps = utils.list_feature_directories(path, "fingerprint")
            rows = []
            for name in sorted(names):
                entry = os.path.join(path, name)
                try:
                    modified = time.ctime(os.path.getmtime(entry))
                except OSError:
                    # A broken link or a file removed during refresh must not hide saved models.
                    continue
                rows.append([name, "Directory" if os.path.isdir(entry) else "File", modified])
            updates = {contents: pd.DataFrame(rows, columns=["Name", "Type", "Modified"])}
            for component, choices, old in zip(selection_components, [csvs, csvs, fps, csvs, fps, models], selected):
                updates[component] = gr.update(choices=choices, value=old if old in choices else (choices[0] if choices else None))
            return updates
        except Exception as exc:
            return {status_html: status(f"Error refreshing directory: {exc}")}

    refresh_outputs = [contents, *selection_components, status_html]
    reset_outputs = list(reset())

    def on_open_working_directory(name):
        message, path = workflow.on_open_working_directory(name)
        updates = reset()
        refreshed = refresh_directory(path, *([None] * len(selection_components)))
        if status_html in refreshed:
            # Do not leave choices from the previous directory active when listing fails.
            updates[contents] = pd.DataFrame(columns=["Name", "Type", "Modified"])
            updates.update({component: gr.update(choices=[], value=None)
                            for component in selection_components})
        updates.update(refreshed)
        updates.update({status_html: refreshed.get(status_html, message), path_state: path,
                        directory: (gr.update(choices=workflow.get_working_directories(), value=os.path.basename(path))
                                    if path else gr.update(value=None)),
                        preview: gr.update(value=None)})
        updates.update({button: gr.update(interactive=path is not None and status_html not in refreshed)
                        for button in (upload_button, extract_button, process_button, load_button)})
        return updates

    open_outputs = [*reset_outputs, *refresh_outputs, path_state, directory, preview,
                    upload_button, extract_button, process_button, load_button]
    # Components may serve more than one role; each output must be listed once.
    open_outputs = list(dict.fromkeys(open_outputs))
    open_button.click(on_open_working_directory, directory, open_outputs, concurrency_id=concurrency_id)
    directory.change(on_open_working_directory, directory, open_outputs, concurrency_id=concurrency_id)
    refresh_button.click(refresh_directory, [path_state, *selection_components], refresh_outputs, concurrency_id=concurrency_id)

    def on_upload_dataset(file_path, path, name, smiles, target_columns, training_source,
                          canonicalize_smiles=True, remove_duplicates=True, remove_salt=True):
        message, frame = workflow.on_upload_dataset(
            file_path, path, name, smiles, target_columns,
            canonicalize_smiles=canonicalize_smiles, remove_duplicates=remove_duplicates, remove_salt=remove_salt)
        updates = {status_html: message, preview: frame}
        if (succeeded(message) and training_source
                and training_source[:2] == (path, f"{str(name).strip()}.csv")):
            updates.update(reset())
        return updates

    upload_button.click(on_upload_dataset, [file_input, path_state, dataset_name, smiles_column, targets, training_source_state,
                                           canonicalize_smiles_checkbox, remove_duplicates_checkbox, remove_salt_checkbox],
                        [status_html, preview, *reset_outputs], concurrency_id=concurrency_id).then(
        refresh_directory, [path_state, *selection_components], refresh_outputs, concurrency_id=concurrency_id)

    def on_extract_fingerprints(path, dataset, fp_name, method, radius_value, bits_value, training_source, progress=gr.Progress()):
        message = workflow.on_extract_fingerprints(path, dataset, fp_name, method, radius_value, bits_value, progress)
        updates = {status_html: message}
        try:
            destination = utils.prefixed_feature_directory(fp_name, "fingerprint")
        except ValueError:
            destination = None
        if (succeeded(message) and training_source and training_source[0] == path
                and (training_source[1] == dataset or training_source[2] == destination)):
            updates.update(reset())
        return updates

    extract_button.click(on_extract_fingerprints,
                         [path_state, extraction_dataset, fingerprint_name, fingerprint_method, radius, bits, training_source_state],
                         [status_html, *reset_outputs], concurrency_id=concurrency_id).then(
        refresh_directory, [path_state, *selection_components], refresh_outputs, concurrency_id=concurrency_id)
    fingerprint_method.change(lambda name: gr.update(visible=name == "Morgan fingerprint"), fingerprint_method, radius)

    def on_process_data(path, dataset, fps, test_size, random_seed, split_method="Random"):
        message, processed = workflow.on_process_data(path, dataset, fps, test_size, random_seed, split_method)
        updates = reset()
        updates.update({status_html: message, processed_state: processed,
                        split_context_state: processed.split_context if processed is not None else None,
                        training_source_state: (path, dataset, fps) if processed is not None else None,
                        fit_button: gr.update(interactive=processed is not None)})
        return updates

    process_button.click(on_process_data,
                         [path_state, training_dataset, training_fps, test_ratio, split_seed, split_method],
                         [status_html, *reset_outputs], concurrency_id=concurrency_id)

    def on_model_change(name, processed):
        updates = reset(clear_processed=False)
        updates.update({group: gr.update(visible=model_name == name) for model_name, group in parameter_groups.items()})
        updates[fit_button] = gr.update(interactive=processed is not None)
        return updates

    model_choice.change(on_model_change, [model_choice, processed_state],
                        [*reset(clear_processed=False), *parameter_groups.values()], concurrency_id=concurrency_id)

    def on_fit(processed, name, random_seed, current_model, user_run_name, *values):
        parameters = {key: value for (model_name, key), value in zip(parameter_keys, values) if model_name == name}
        message, fitted, summary = workflow.on_fit(processed, name, parameters, random_seed, current_model,
                                                  run_name=user_run_name)
        updates = {status_html: message, model_state: fitted}
        if succeeded(message):
            updates.update(reset(clear_processed=False))
            updates.update({model_state: fitted, fitted_summary: summary, fit_button: gr.update(interactive=True),
                            evaluate_button: gr.update(interactive=True), predict_button: gr.update(interactive=True),
                            save_button: gr.update(interactive=True)})
        return updates

    fit_button.click(on_fit, [processed_state, model_choice, model_seed, model_state, run_name, *parameter_components],
                     [status_html, *reset(clear_processed=False)], concurrency_id=concurrency_id)

    def target_metrics(metrics, target):
        return metrics.loc[metrics["Target"] == target].reset_index(drop=True)

    def on_evaluate(model, processed, previous_metrics, previous_rows, target):
        message, metrics, rows, selected, plot_one, plot_two = workflow.on_evaluate(
            model, processed, previous_metrics, previous_rows, target)
        updates = {status_html: message, test_state: metrics, evaluation_rows_state: rows}
        if succeeded(message):
            updates.update({test_table: target_metrics(metrics, selected),
                            output_target: gr.update(choices=list(model.target_names), value=selected, interactive=True),
                            export_test: gr.update(interactive=True), export_evaluation: gr.update(interactive=True)})
            updates.update(zip(evaluation_plots, [plot_one, plot_two]))
        return updates

    evaluate_button.click(on_evaluate, [model_state, processed_state, test_state, evaluation_rows_state, output_target],
                          [status_html, test_state, evaluation_rows_state, test_table, output_target,
                           export_test, export_evaluation, *evaluation_plots], concurrency_id=concurrency_id)

    def on_output_change(target, rows, metrics):
        if target is None:
            return {test_table: gr.update()}
        try:
            if rows is None or metrics is None:
                raise ValueError("Please evaluate the model first.")
            selected_metrics = target_metrics(metrics, target)
            if selected_metrics.empty:
                raise ValueError("Please select an evaluated output target.")
            plot_one, plot_two = make_evaluation_plots(workflow.task, rows, target)
            updates = {test_table: selected_metrics}
            updates.update(zip(evaluation_plots, [plot_one, plot_two]))
            return updates
        except Exception as exc:
            return {status_html: status(f"Error displaying evaluation: {exc}")}

    output_target.change(on_output_change, [output_target, evaluation_rows_state, test_state],
                         [status_html, test_table, *evaluation_plots], concurrency_id=concurrency_id)

    def on_predict(model, path, dataset, fps, previous):
        message, predictions = workflow.on_predict(model, path, dataset, fps, previous)
        return {status_html: message, prediction_state: predictions,
                prediction_table: predictions if succeeded(message) else gr.update(),
                export_prediction: gr.update(interactive=predictions is not None and not predictions.empty)}

    predict_button.click(on_predict, [model_state, path_state, prediction_dataset, prediction_fps, prediction_state],
                         [status_html, prediction_state, prediction_table, export_prediction], concurrency_id=concurrency_id)

    def on_load_model(path, filename, current, processed, user_run_name=None):
        message, fitted, summary = workflow.on_load_model(path, filename, current, processed, user_run_name)
        updates = {status_html: message, model_state: fitted}
        if succeeded(message):
            updates.update(reset(clear_processed=False))
            updates.update({model_state: fitted, fitted_summary: summary,
                            fit_button: gr.update(interactive=processed is not None),
                            evaluate_button: gr.update(interactive=processed is not None),
                            predict_button: gr.update(interactive=True), save_button: gr.update(interactive=True)})
            if fitted.run_context is not None:
                updates.update({run_name: gr.update(value=fitted.run_context["model_name"]),
                                model_seed: gr.update(value=fitted.random_seed)})
        return updates

    load_button.click(on_load_model, [path_state, saved_model, model_state, processed_state, run_name],
                      [status_html, *reset(clear_processed=False), run_name, model_seed], concurrency_id=concurrency_id)

    def on_save_model(path, name, fitted, *selected):
        message = workflow.on_save_model(path, name, fitted)
        updates = {status_html: message}
        if succeeded(message):
            name = utils.sanitize_path_component(name, "model file name")
            filename = name if name.endswith(".joblib") else f"{name}.joblib"
            relative = f"{os.path.basename(run_utils.run_directory(fitted.run_context))}/{filename}"
            selections = list(selected) if selected else [None] * len(selection_components)
            selections[-1] = relative
            # Return the saved file and its refreshed choices in the save response itself.
            updates.update(refresh_directory(path, *selections))
        return updates

    save_button.click(on_save_model, [path_state, save_name, model_state, *selection_components],
                      refresh_outputs, concurrency_id=concurrency_id)
    for button, name, state in zip(export_buttons, [test_name, evaluation_name, prediction_name], result_states):
        button.click(workflow.on_export, [path_state, name, state, model_state], status_html, concurrency_id=concurrency_id).then(
            refresh_directory, [path_state, *selection_components], refresh_outputs, concurrency_id=concurrency_id)
    bind_similarity_controls(similarity_controls, split_context_state, status_html, concurrency_id=concurrency_id)
    return tab
