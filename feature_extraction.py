"""Shared graph/fingerprint extraction controls and safe action responses."""

from dataclasses import dataclass
from html import escape
import os

import gradio as gr

from update_dataset import directory_contents
import utils


GRAPH_FEATURIZERS = ["MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer"]


@dataclass
class FeatureExtractionControls:
    dataset: gr.Dropdown
    graph_directory: gr.Textbox
    graph_featurizer: gr.Dropdown | None
    graph_button: gr.Button
    fingerprint_directory: gr.Textbox | None = None
    fingerprint_method: gr.Dropdown | None = None
    radius: gr.Slider | None = None
    bits: gr.Slider | None = None
    fingerprint_button: gr.Button | None = None
    number_of_conformers: gr.Slider | None = None
    force_field: gr.Dropdown | None = None


def build_feature_extraction_controls(is_3d=False):
    """Build matching controls for 2D graphs/fingerprints or 3D conformers."""
    with gr.Accordion("Feature Extraction"):
        with gr.Row():
            dataset = gr.Dropdown(label="Dataset file", choices=[], allow_custom_value=False)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**3D graph extraction**" if is_3d else "**Graph extraction**")
                graph_directory = gr.Textbox(
                    label="Graph directory", value="GRAPH_graphs_3d" if is_3d else "GRAPH_graphs",
                    info="GRAPH_ is added automatically when omitted.")
                if is_3d:
                    graph_featurizer = None
                    number_of_conformers = gr.Slider(label="Number of conformers", minimum=1,
                                                    maximum=100, value=30, step=1)
                    force_field = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"],
                                              allow_custom_value=False)
                else:
                    graph_featurizer = gr.Dropdown(label="Graph featurizer", value=GRAPH_FEATURIZERS[0],
                                                  choices=GRAPH_FEATURIZERS)
                    number_of_conformers = force_field = None
                graph_button = gr.Button(value="Extract graphs", interactive=False)
            fingerprint_directory = fingerprint_method = radius = bits = fingerprint_button = None
            if not is_3d:
                with gr.Column(scale=1):
                    gr.Markdown("**Molecular fingerprint extraction**")
                    fingerprint_directory = gr.Textbox(
                        label="Molecular fingerprint directory", value="FP_fingerprints",
                        info="FP_ is added automatically when omitted. Use a separate folder for each dataset.")
                    fingerprint_method = gr.Dropdown(label="Molecular fingerprint", value="Morgan fingerprint",
                                                     choices=utils.FINGERPRINT_NAMES)
                    radius = gr.Slider(label="Radius", value=2, minimum=1, maximum=5, step=1)
                    bits = gr.Slider(label="Number of bits", value=2048, minimum=128, maximum=4096, step=128)
                    fingerprint_button = gr.Button(value="Extract molecular fingerprints", interactive=False)
    return FeatureExtractionControls(dataset, graph_directory, graph_featurizer, graph_button,
                                     fingerprint_directory, fingerprint_method, radius, bits,
                                     fingerprint_button, number_of_conformers, force_field)


def fingerprint_radius_update(name):
    return gr.update(visible=name == "Morgan fingerprint")


def _run_extraction(extractor, working_directory_path, *args):
    try:
        if not working_directory_path or not os.path.isdir(working_directory_path):
            raise ValueError("Please open a working directory first.")
        message = extractor(working_directory_path, *args)
        color = "green" if "color: green" in message else "red"
        inner = message.split(">", 1)[1].rsplit("</span>", 1)[0] if message.startswith("<span") else message
        status = f"<span style='color: {color};'>{escape(inner)}</span>"
    except Exception as exc:
        status = f"<span style='color: red;'>Error extracting features: {escape(str(exc))}</span>"
    return status, directory_contents(working_directory_path)


def on_extract_graphs(working_directory_path, dataset_file_name, graph_directory, graph_featurizer_name,
                      datatype, progress=gr.Progress()):
    return _run_extraction(utils.extract_graphs, working_directory_path, dataset_file_name,
                           graph_directory, graph_featurizer_name, datatype, progress)


def on_extract_3d_graphs(working_directory_path, dataset_file_name, graph_directory,
                         number_of_conformers, force_field, datatype, progress=gr.Progress()):
    return _run_extraction(utils.extract_3d_graphs, working_directory_path, dataset_file_name,
                           graph_directory, number_of_conformers, force_field, datatype, progress)


def on_extract_fingerprints(working_directory_path, dataset_file_name, molecular_fingerprint_directory,
                            molecular_fingerprint_name, radius, number_of_bits, progress=gr.Progress()):
    return _run_extraction(utils.extract_molecule_fingerprints, working_directory_path, dataset_file_name,
                           molecular_fingerprint_directory, molecular_fingerprint_name, radius,
                           number_of_bits, progress)
