"""Dataset upload controls and import responses shared by all six workflows."""

from dataclasses import dataclass
from html import escape
import os

import gradio as gr

import utils


@dataclass
class DatasetUploadControls:
    file: gr.File
    name: gr.Textbox
    smiles: gr.Textbox
    targets: gr.Textbox
    canonicalize_smiles: gr.Checkbox
    remove_duplicates: gr.Checkbox
    remove_salt: gr.Checkbox
    button: gr.Button
    preview: gr.Dataframe


def dataset_cleanup_controls():
    """Keep import options together immediately above the upload action."""
    with gr.Row():
        canonicalize_smiles = gr.Checkbox(
            label="Canonicalize SMILES", value=True, min_width=0,
            info="Standardize SMILES while preserving stereochemistry.")
        remove_duplicates = gr.Checkbox(
            label="Remove duplicates", value=True, min_width=0,
            info="Keep the first row for each molecule, including its target values.")
        remove_salt = gr.Checkbox(
            label="Remove salt", value=True, min_width=0,
            info="Strip recognized salt fragments from the imported SMILES.")
    return canonicalize_smiles, remove_duplicates, remove_salt


def build_dataset_upload_controls(is_3d=False):
    """Build the common upload form, retaining the 3D target restriction."""
    with gr.Accordion("Upload Dataset"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                file = gr.File(file_types=[".csv"], type="filepath", label="Dataset file")
                name = gr.Textbox(label="Dataset name", value="dataset",
                                  info="Use a new name when uploading prediction data.")
                smiles = gr.Textbox(label="SMILES column", value="SMILES")
                targets = gr.Textbox(
                    label="Target columns (separated by commas)", value="target",
                    info=("3D models accept one target. Leave blank for prediction data." if is_3d else
                          "Leave blank when uploading an unlabeled prediction dataset."))
                canonicalize_smiles, remove_duplicates, remove_salt = dataset_cleanup_controls()
                button = gr.Button(value="Load dataset", interactive=False)
            with gr.Column(scale=2):
                preview = gr.Dataframe(label="Dataset preview", max_height=360, interactive=False)
    return DatasetUploadControls(file, name, smiles, targets, canonicalize_smiles,
                                 remove_duplicates, remove_salt, button, preview)


def directory_contents(path):
    """Return a safe root listing for an action's directory-refresh state."""
    if not path:
        return []
    try:
        return os.listdir(path)
    except OSError:
        return []


def upload_dataset(dataset_file_path, working_directory_path, dataset_name, smiles_column,
                   target_columns, canonicalize_smiles=True, remove_duplicates=True,
                   remove_salt=True, *, is_3d_dataset=False):
    """Import once and preserve displayed data if validation or publication fails."""
    try:
        if not working_directory_path or not os.path.isdir(working_directory_path):
            raise ValueError("Please open a working directory first.")
        if dataset_file_path is None or not str(dataset_file_path).strip():
            raise ValueError("Please select a dataset file.")
        options = dict(canonicalize_smiles=canonicalize_smiles,
                       remove_duplicates=remove_duplicates, remove_salt=remove_salt)
        if is_3d_dataset:
            options["is_3d_dataset"] = True
        status, frame = utils.process_dataset_file(
            dataset_file_path, working_directory_path, dataset_name, smiles_column,
            target_columns, **options)
        if frame is None:
            return status, gr.update(), gr.update()
        return status, frame.head(10), directory_contents(working_directory_path)
    except Exception as exc:
        return (f"<span style='color: red;'>Error uploading dataset: {escape(str(exc))}</span>",
                gr.update(), gr.update())
