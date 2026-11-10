"""Release disconnected browser sessions without mutating active callbacks' inputs."""

import logging

import gradio as gr


def detach_session(blocks, session_hash):
    """Remove the holder's ownership of one session, including rendered payloads.

    Gradio's ordinary state expiry retains the surrounding session and component
    values. This small adapter deliberately detaches the whole SessionState instead
    of clearing its dictionaries: running callbacks already hold that object and may
    safely finish postprocessing into it. Their local references then expire without
    putting the disconnected session back in the holder.

    There are no worker shutdowns or model mutations here. In particular, resources
    still used by an in-flight training action remain alive until that action ends.
    """
    if not session_hash:
        return False
    holder = getattr(blocks, "state_holder", None)
    if holder is None or not all(hasattr(holder, name) for name in
                                 ("lock", "session_data", "time_last_used")):
        logging.getLogger(__name__).warning(
            "Session cleanup is unavailable with this Gradio version; using its normal expiry.")
        return False
    with holder.lock:
        session = holder.session_data.pop(session_hash, None)
        holder.time_last_used.pop(session_hash, None)
    return session is not None


def register_session_cleanup(blocks):
    """Use Gradio's disconnect event for both launch() and mounted FastAPI apps."""
    async def on_session_unload(request: gr.Request):
        if request is None or not request.session_hash:
            return
        # The heartbeat and queue SSE connections can close in either order.
        # Remove pending jobs before detaching their state so an old queued action
        # cannot start with a new empty session. This is Gradio's own disconnect
        # cleanup; it flags active jobs but does not cancel their running callbacks.
        queue = getattr(blocks, "_queue", None)
        clean_events = getattr(queue, "clean_events", None)
        if clean_events is None:
            logging.getLogger(__name__).warning(
                "Queue cleanup is unavailable with this Gradio version; using its normal expiry.")
            return
        await clean_events(session_hash=request.session_hash)
        detach_session(blocks, request.session_hash)

    blocks.unload(on_session_unload)
