"""Fingerprint-based classical binary classification tab."""

from classical_ui import ClassicalWorkflow, build_classical_tab
from utils import DATA_DIR


_workflow = ClassicalWorkflow("classification", lambda: DATA_DIR)


def get_working_directories():
    return _workflow.get_working_directories()


def on_open_working_directory(name):
    return _workflow.on_open_working_directory(name)


def binary_classification_classical_tab_content():
    return build_classical_tab(_workflow, "Binary Classification - Classical ML Models")
