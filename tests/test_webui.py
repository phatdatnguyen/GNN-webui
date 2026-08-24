"""Tests for app wiring: the Gradio Blocks tree and the port reservation helper."""
import socket

import gradio as gr
import pytest

import binary_classification as bc
import binary_classification_3d as bc3
import regression as rg
import regression_3d as rg3


TAB_BUILDERS = [
    ("Binary Classification", bc.binary_classification_tab_content),
    ("Binary Classification 3D", bc3.binary_classification_3d_tab_content),
    ("Regression", rg.regression_tab_content),
    ("Regression 3D", rg3.regression_3d_tab_content),
]


@pytest.mark.parametrize("label,builder", TAB_BUILDERS, ids=[n for n, _ in TAB_BUILDERS])
def test_each_tab_builds(label, builder):
    """A mismatch between a callback's return count and its outputs list only shows up
    when the Blocks tree is constructed."""
    with gr.Blocks():
        tab = builder()
    assert tab.label == label


def test_all_tabs_build_together():
    with gr.Blocks() as blocks:
        with gr.Tabs():
            for _, builder in TAB_BUILDERS:
                builder()
    assert len(blocks.blocks) > 100


def test_reserve_available_port_returns_a_listening_socket():
    import webui

    sock, port = webui.reserve_available_port()
    try:
        assert 7860 <= port < 7960
        assert sock.getsockname()[1] == port
        # The socket is held open, so nothing else can claim the port in the meantime.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
            with pytest.raises(OSError):
                other.bind(("127.0.0.1", port))
    finally:
        sock.close()


def test_reserve_available_port_skips_a_busy_port():
    import webui

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 7860))
        blocker.listen(1)
        sock, port = webui.reserve_available_port(7860)
        try:
            assert port > 7860
        finally:
            sock.close()


def test_reserve_available_port_gives_up_eventually():
    import webui

    with pytest.raises(RuntimeError):
        webui.reserve_available_port(1, max_attempts=1)     # port 1 needs root
