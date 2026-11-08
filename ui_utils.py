"""Shared Gradio wiring for dependent session state."""

from functools import wraps

import gradio as gr


def dataset_cleanup_controls():
    """Consistent dataset import options for each workflow."""
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


def bind_result_invalidation(button, callback, inputs, outputs, *, clear_states,
                             disable_buttons, clear_components, concurrency_id=None):
    """Clear cached results in the same response as a model-changing action.

    Even a failed training call can update weights before reporting an error, and
    model creation can discard the previous model. Neither may leave old exports
    available as if they described the current model.
    """
    all_outputs = [*outputs, *clear_states, *disable_buttons, *clear_components]
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("Model action outputs and result resets must not overlap.")

    @wraps(callback)
    def change_and_reset(*args, **kwargs):
        result = dict(zip(outputs, callback(*args, **kwargs), strict=True))
        result.update((state, None) for state in clear_states)
        result.update((component, gr.update(interactive=False)) for component in disable_buttons)
        result.update((component, gr.update(value=None)) for component in clear_components)
        return result

    return button.click(change_and_reset, inputs, all_outputs, concurrency_id=concurrency_id)


def bind_data_processing(button, callback, inputs, outputs, *, clear_states,
                         loss_states, epoch_state, disable_buttons, clear_components,
                         concurrency_id=None):
    """Publish processed data and invalidate its old model in one callback response.

    The original handler keeps its positional API for direct callers. The UI wrapper
    resets dependent state on success and failure, since either result replaces the
    current dataset/loaders. Keeping resets in this callback avoids a separate event
    that could expose new data alongside an old model or fitted target scaler.
    """
    all_outputs = [*outputs, *clear_states, *loss_states, epoch_state,
                   *disable_buttons, *clear_components]
    if len(all_outputs) != len(set(all_outputs)):
        raise ValueError("Data processing outputs and reset components must not overlap.")

    @wraps(callback)
    def process_and_reset(*args, **kwargs):
        result = dict(zip(outputs, callback(*args, **kwargs), strict=True))
        result.update((state, None) for state in clear_states)
        result.update((state, []) for state in loss_states)
        result[epoch_state] = 0
        result.update((component, gr.update(interactive=False)) for component in disable_buttons)
        result.update((component, gr.update(value=None)) for component in clear_components)
        return result

    return button.click(process_and_reset, inputs, all_outputs, concurrency_id=concurrency_id)
