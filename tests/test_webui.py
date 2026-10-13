"""Tests for app wiring: the Gradio Blocks tree and the port reservation helper."""
import socket

import gradio as gr
import pytest

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import binary_classification_classical as bcc
import regression_hybrid as rg
import regression_3d as rg3
import regression_classical as rgc


TAB_BUILDERS = [
    ("Binary Classification - Hybrid GNN", bc.binary_classification_hybrid_tab_content),
    ("Binary Classification - 3D GNN", bc3.binary_classification_3d_tab_content),
    ("Binary Classification - Classical ML Models", bcc.binary_classification_classical_tab_content),
    ("Regression - Hybrid GNN", rg.regression_hybrid_tab_content),
    ("Regression - 3D GNN", rg3.regression_3d_tab_content),
    ("Regression - Classical ML Models", rgc.regression_classical_tab_content),
]


def _ancestor(component, component_type):
    parent = component.parent
    while parent is not None:
        if isinstance(parent, component_type):
            return parent
        parent = parent.parent
    return None


def _inside(component, container):
    parent = component.parent
    while parent is not None:
        if parent is container:
            return True
        parent = parent.parent
    return False


def _two_columns(accordion):
    rows = [child for child in accordion.children if isinstance(child, gr.Row)]
    assert len(rows) == 1
    columns = [child for child in rows[0].children if isinstance(child, gr.Column)]
    assert [column.scale for column in columns] == [1, 2]
    return columns


@pytest.mark.parametrize("label,builder", TAB_BUILDERS, ids=[n for n, _ in TAB_BUILDERS])
def test_each_tab_builds(label, builder):
    """Validate component structure and wiring; API tests check runtime return values."""
    with gr.Blocks() as blocks:
        tab = builder()
    assert tab.label == label
    split_radios = [component for component in blocks.blocks.values()
                    if isinstance(component, gr.Radio) and component.label == "Split method"]
    assert len(split_radios) == 1
    split_radio = split_radios[0]
    assert split_radio.value == "Random"
    assert [value for _, value in split_radio.choices] == ["Random", "Scaffold"]
    process_callbacks = [callback for callback in blocks.fns.values()
                         if getattr(callback.fn, "__name__", None) == "on_process_data"]
    assert len(process_callbacks) == 1
    assert process_callbacks[0].inputs[-1] is split_radio

    accordions = [component for component in blocks.blocks.values() if isinstance(component, gr.Accordion)]
    for title in ("Model evaluation", "Make predictions"):
        assert sum(accordion.label == title for accordion in accordions) == 1
    assert not any(accordion.label in ("Make prediction", "Test evaluation") for accordion in accordions)

    evaluation = next(accordion for accordion in accordions if accordion.label == "Model evaluation")
    controls, plots_column = _two_columns(evaluation)
    evaluation_plots = [component for component in blocks.blocks.values()
                        if isinstance(component, gr.Plot) and _inside(component, evaluation)]
    assert len(evaluation_plots) == (2 if label.startswith("Binary Classification") else 1)
    assert all(_ancestor(plot, gr.Column) is plots_column for plot in evaluation_plots)
    metrics = [component for component in blocks.blocks.values()
               if isinstance(component, gr.Dataframe) and _inside(component, evaluation)]
    assert len(metrics) == 1
    assert _ancestor(metrics[0], gr.Column) is controls
    export_fields = {component.label: component for component in blocks.blocks.values()
                     if isinstance(component, gr.Textbox) and _inside(component, evaluation)}
    assert export_fields["Metrics export file"].value == "test_metrics"
    assert export_fields["Evaluation export file"].value == "evaluation"
    export_buttons = [component for component in blocks.blocks.values()
                      if isinstance(component, gr.Button) and _inside(component, evaluation)
                      and component.value.startswith("Export")]
    assert [button.value for button in export_buttons] == [
        "Export test metrics (all targets)", "Export evaluation rows (all targets)"]
    assert all(button.interactive is False for button in export_buttons)
    for field in export_fields.values():
        assert _ancestor(field, gr.Column) is controls

    prediction = next(accordion for accordion in accordions if accordion.label == "Make predictions")
    inputs_column, results_column = _two_columns(prediction)
    prediction_inputs = [component for component in blocks.blocks.values()
                         if isinstance(component, gr.Dropdown) and component.label == "Prediction dataset file"
                         and _inside(component, prediction)]
    assert len(prediction_inputs) == 1
    assert _ancestor(prediction_inputs[0], gr.Column) is inputs_column
    prediction_tables = [component for component in blocks.blocks.values()
                         if isinstance(component, gr.Dataframe) and _inside(component, prediction)]
    assert len(prediction_tables) == 1
    assert _ancestor(prediction_tables[0], gr.Column) is results_column

    if "Classical ML Models" in label:
        assert not any(getattr(component, "label", None) in
                       ("Validation size", "Validation metrics", "Validation export file")
                       for component in blocks.blocks.values())
        test_sliders = [component for component in blocks.blocks.values()
                        if isinstance(component, gr.Slider) and component.label == "Test size"]
        assert len(test_sliders) == 1
        assert test_sliders[0].value == 0.2
        model_accordion = next(accordion for accordion in accordions if accordion.label == "Classical ML model")
        model_choices = [component for component in blocks.blocks.values()
                         if isinstance(component, gr.Dropdown) and component.label == "Model"
                         and _inside(component, model_accordion)]
        assert len(model_choices) == 1
        selector_column = _ancestor(model_choices[0], gr.Column)
        assert selector_column.scale == 1
        parameter_groups = [component for component in blocks.blocks.values()
                            if isinstance(component, gr.Group) and _inside(component, model_accordion)]
        assert len(parameter_groups) == len(model_choices[0].choices)
        assert sum(group.visible for group in parameter_groups) == 1
        parameter_columns = {_ancestor(group, gr.Column) for group in parameter_groups}
        assert len(parameter_columns) == 1
        parameter_column = parameter_columns.pop()
        assert parameter_column.scale == 2
        assert isinstance(selector_column.parent, gr.Row)
        assert parameter_column.parent is selector_column.parent
        assert selector_column.parent.children.index(selector_column) < selector_column.parent.children.index(parameter_column)
        expected_plot_labels = (["Confusion matrix", "ROC curve"]
                                if label.startswith("Binary Classification") else ["Actual vs. predicted"])
        assert [plot.label for plot in evaluation_plots] == expected_plot_labels


def test_all_tabs_build_together():
    with gr.Blocks() as blocks:
        with gr.Tabs():
            for _, builder in TAB_BUILDERS:
                builder()
    assert len(blocks.blocks) > 100


@pytest.fixture
def free_port():
    """Ask the OS for an unused port, then release it.

    The tests below must not hardcode 7860 - a developer may well have something else
    (including another copy of this app) already listening there.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_reserve_available_port_returns_a_listening_socket(free_port):
    import webui

    sock, port = webui.reserve_available_port(free_port)
    try:
        assert port >= free_port
        assert sock.getsockname()[1] == port
        # The socket is held open, so nothing else can claim the port in the meantime.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
            with pytest.raises(OSError):
                other.bind(("127.0.0.1", port))
    finally:
        sock.close()


def test_reserve_available_port_skips_a_busy_port(free_port):
    import webui

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", free_port))
        blocker.listen(1)
        sock, port = webui.reserve_available_port(free_port)
        try:
            assert port > free_port
        finally:
            sock.close()


def test_reserve_available_port_gives_up_eventually():
    import webui

    with pytest.raises(RuntimeError):
        webui.reserve_available_port(1, max_attempts=1)     # port 1 needs root
