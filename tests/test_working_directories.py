"""Classification and regression working directories stay within their workflow roots."""
import pytest

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3


@pytest.fixture(params=[(bc, "Classification"), (bc3, "Classification"),
                       (rg, "Regression"), (rg3, "Regression")],
                ids=["classification", "classification_3d", "regression", "regression_3d"])
def workflow(request, tmp_path, monkeypatch):
    module, name = request.param
    data_dir = tmp_path / "Data"
    monkeypatch.setattr(module, "DATA_DIR", str(data_dir))
    return module, data_dir, data_dir / name


def test_listing_creates_workflow_root(workflow):
    module, data_dir, workflow_root = workflow

    assert module.get_working_directories() == []
    assert workflow_root.is_dir()
    assert list(data_dir.iterdir()) == [workflow_root]


def test_new_working_directory_is_created_inside_workflow(workflow):
    module, data_dir, workflow_root = workflow

    result = module.on_open_working_directory("new_run")

    assert result[-2] == str(workflow_root / "new_run")
    assert (workflow_root / "new_run").is_dir()
    assert not (data_dir / "new_run").exists()
    assert result[-1] == []
    assert result[0]["choices"] == ["new_run"]
    assert result[0]["value"] == "new_run"
    assert all(update["interactive"] for update in result[1:-2])


def test_listing_is_sorted_and_scoped_to_own_workflow(workflow):
    module, data_dir, workflow_root = workflow
    other_root = data_dir / ("Regression" if workflow_root.name == "Classification" else "Classification")
    for directory in [workflow_root / "z_run", workflow_root / "a_run",
                      other_root / "other_run", data_dir / "legacy_run"]:
        directory.mkdir(parents=True)
    (workflow_root / "notes.txt").write_text("not a working directory")

    assert module.get_working_directories() == ["a_run", "z_run"]


def test_opening_existing_working_directory_preserves_files(workflow):
    module, _, workflow_root = workflow
    working_directory = workflow_root / "existing"
    graph_directory = working_directory / "GRAPH_graphs"
    graph_directory.mkdir(parents=True)
    (working_directory / "dataset.csv").write_text("SMILES,target\nCCO,1\n")
    (graph_directory / "0.pt").write_bytes(b"existing graph")

    result = module.on_open_working_directory("existing")

    assert result[-2] == str(working_directory)
    assert sorted(result[-1]) == ["GRAPH_graphs", "dataset.csv"]
    assert (working_directory / "dataset.csv").read_text() == "SMILES,target\nCCO,1\n"
    assert (graph_directory / "0.pt").read_bytes() == b"existing graph"


def test_working_directory_traversal_is_rejected(workflow):
    module, data_dir, _ = workflow

    result = module.on_open_working_directory("../escape")

    assert result[-2] is None
    assert all(update["interactive"] is False for update in result[1:-2])
    assert not data_dir.exists()
    assert not (data_dir.parent / "escape").exists()


@pytest.mark.parametrize("two_d,three_d,workflow_name", [
    (bc, bc3, "Classification"), (rg, rg3, "Regression"),
])
def test_two_dimensional_and_three_dimensional_tabs_share_their_workflow_root(
        tmp_path, monkeypatch, two_d, three_d, workflow_name):
    data_dir = tmp_path / "Data"
    for module in [two_d, three_d]:
        monkeypatch.setattr(module, "DATA_DIR", str(data_dir))

    two_d.on_open_working_directory("shared")
    assert three_d.get_working_directories() == ["shared"]
    assert three_d.on_open_working_directory("shared")[-2] == str(data_dir / workflow_name / "shared")
