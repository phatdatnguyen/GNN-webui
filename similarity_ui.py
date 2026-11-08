"""Shared train–test similarity controls for all six model workflows."""

from dataclasses import dataclass
import html

import gradio as gr
from matplotlib import pyplot as plt
import numpy as np

from similarity import calculate_similarity, export_similarity


@dataclass
class SimilarityControls:
    state: gr.State
    radius: gr.Number
    number_of_bits: gr.Dropdown
    calculate_button: gr.Button
    pair_plot: gr.Plot
    nearest_plot: gr.Plot
    summary: gr.Textbox
    preview: gr.Dataframe
    distribution_name: gr.Textbox
    nearest_name: gr.Textbox
    distribution_export_button: gr.Button
    nearest_export_button: gr.Button

    def reset_updates(self):
        return {self.state: None, self.pair_plot: gr.update(value=None),
                self.nearest_plot: gr.update(value=None), self.summary: gr.update(value=None),
                self.preview: gr.update(value=None),
                self.distribution_export_button: gr.update(interactive=False),
                self.nearest_export_button: gr.update(interactive=False)}

    @property
    def reset_outputs(self):
        return list(self.reset_updates())


def build_similarity_controls():
    with gr.Group():
        gr.Markdown("### Training–test similarity")
        gr.Markdown("Compare the processed training and test molecules using Morgan fingerprints and Tanimoto similarity. "
                    "Validation molecules are excluded. Recognized salts are stripped and explicit hydrogens are added, "
                    "matching feature extraction; chirality is ignored. Every training–test pair is compared, so large splits take longer.")
        state = gr.State()
        with gr.Row():
            radius = gr.Number(label="Similarity Morgan radius", value=2, minimum=0, maximum=16, precision=0)
            number_of_bits = gr.Dropdown(label="Similarity fingerprint bits", value=2048,
                                        choices=[128, 256, 512, 1024, 2048, 4096])
        calculate_button = gr.Button("Calculate training–test similarity")
        with gr.Row():
            pair_plot = gr.Plot(label="All training–test pair similarities")
            nearest_plot = gr.Plot(label="Nearest training similarity per test molecule")
        summary = gr.Textbox(label="Similarity summary", lines=4, interactive=False)
        preview = gr.Dataframe(label="Nearest training molecules (first 20 test rows)", interactive=False)
        gr.Markdown("Row IDs are zero-based positions in the processed dataset. Ties choose the lowest training row ID. "
                    "CSV exports include all rows and a metadata file, saved under the captured data-splitting seed.")
        with gr.Row():
            with gr.Column():
                distribution_name = gr.Textbox(label="Similarity distribution export file", value="similarity_distribution")
                distribution_export_button = gr.Button("Export similarity distribution", interactive=False)
            with gr.Column():
                nearest_name = gr.Textbox(label="Nearest training molecules export file", value="nearest_training_molecules")
                nearest_export_button = gr.Button("Export nearest training molecules", interactive=False)
    return SimilarityControls(state, radius, number_of_bits, calculate_button, pair_plot, nearest_plot,
                              summary, preview, distribution_name, nearest_name,
                              distribution_export_button, nearest_export_button)


def _plots(result):
    figures = []
    try:
        pair_figure, axes = plt.subplots(figsize=(7, 4))
        figures.append(pair_figure)
        distribution = result.distribution
        axes.bar(distribution["Bin lower"], distribution["Proportion"],
                 width=distribution["Bin upper"] - distribution["Bin lower"], align="edge")
        axes.set(xlabel="Tanimoto similarity", ylabel="Proportion of training–test pairs",
                 xlim=(0, 1), title="All training–test pairs")
        pair_figure.tight_layout()
        nearest_figure, axes = plt.subplots(figsize=(7, 4))
        figures.append(nearest_figure)
        nearest_values = result.nearest["Tanimoto similarity"].to_numpy()
        axes.hist(nearest_values, bins=np.linspace(0, 1, result.settings["histogram_bins"] + 1),
                  weights=np.full(len(nearest_values), 1 / len(nearest_values)))
        axes.set(xlabel="Highest training Tanimoto similarity", ylabel="Proportion of test molecules",
                 xlim=(0, 1), title="Nearest training molecule for each test molecule")
        nearest_figure.tight_layout()
        return pair_figure, nearest_figure
    finally:
        for figure in figures:
            plt.close(figure)


def _summary(result):
    summary = result.summary
    return (f"Data splitting seed: {result.context['split_seed']} ({result.context['split_method']}); "
            f"Morgan radius {result.settings['radius']}, {result.settings['number_of_bits']} bits.\n"
            f"Training: {summary['training_molecules']:,}; test: {summary['test_molecules']:,}; "
            f"validation excluded: {summary['validation_molecules_excluded']:,}.\n"
            f"All {summary['pair_count']:,} pairs: mean similarity {summary['mean_pair_similarity']:.4f}.\n"
            f"Nearest training similarity: mean {summary['mean_nearest_similarity']:.4f}, "
            f"median {summary['median_nearest_similarity']:.4f}, "
            f"range {summary['minimum_nearest_similarity']:.4f}–{summary['maximum_nearest_similarity']:.4f}.")


def bind_similarity_controls(bundle, split_context_state, status_html, *, concurrency_id=None):
    concurrency_id = concurrency_id or f"similarity-{bundle.state._id}"

    def on_calculate_similarity(context, radius, number_of_bits, progress=gr.Progress()):
        try:
            result = calculate_similarity(context, radius, number_of_bits, progress=progress)
            pair_figure, nearest_figure = _plots(result)
            return {status_html: "<span style='color: green;'>Training–test similarity calculated.</span>",
                    bundle.state: result, bundle.pair_plot: pair_figure, bundle.nearest_plot: nearest_figure,
                    bundle.summary: _summary(result), bundle.preview: result.nearest.head(20),
                    bundle.distribution_export_button: gr.update(interactive=True),
                    bundle.nearest_export_button: gr.update(interactive=True)}
        except Exception as exc:
            return {status_html: f"<span style='color: red;'>Error calculating similarity: {html.escape(str(exc))}</span>"}

    calculation = bundle.calculate_button.click(
        on_calculate_similarity, [split_context_state, bundle.radius, bundle.number_of_bits],
        [status_html, *bundle.reset_outputs], concurrency_id=concurrency_id)

    def export(result, context, filename, kind):
        try:
            if context is None:
                raise ValueError("Please process the data and calculate similarity before exporting.")
            path, metadata_path = export_similarity(result, kind, filename, current_context=context)
            message = f"Similarity exported to {path}; metadata: {metadata_path}."
            return f"<span style='color: green;'>{html.escape(message)}</span>"
        except Exception as exc:
            return f"<span style='color: red;'>Error exporting similarity: {html.escape(str(exc))}</span>"

    def on_export_similarity_distribution(result, context, filename):
        return export(result, context, filename, "distribution")

    def on_export_nearest_training_molecules(result, context, filename):
        return export(result, context, filename, "nearest")

    distribution_export = bundle.distribution_export_button.click(
        on_export_similarity_distribution, [bundle.state, split_context_state, bundle.distribution_name],
        status_html, concurrency_id=concurrency_id)
    nearest_export = bundle.nearest_export_button.click(
        on_export_nearest_training_molecules, [bundle.state, split_context_state, bundle.nearest_name],
        status_html, concurrency_id=concurrency_id)
    return calculation, distribution_export, nearest_export
