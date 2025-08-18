"""Streamlit UI for CSVAI."""
import asyncio
import hashlib
import logging
import threading
import time
import queue
from pathlib import Path
import tempfile

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from csvai.processor import CSVAIProcessor, ProcessorConfig
from csvai.io_utils import default_output_file

# -----------------------------------------------------------------------------
# Page setup & persistent state
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CSVAI", layout="centered")
st.title("CSVAI")

# Persist across reruns
st.session_state.setdefault("processor", None)
st.session_state.setdefault("thread", None)
st.session_state.setdefault("log_queue", queue.Queue())
st.session_state.setdefault("log_handler_attached", False)
st.session_state.setdefault("raw_logs", [])
st.session_state.setdefault("working_dir", None)
st.session_state.setdefault("output_path", None)

# -----------------------------------------------------------------------------
# Logging: worker -> queue  (no Streamlit calls in worker thread)
# -----------------------------------------------------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self.q = q
    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass  # never raise from logging

def drain_logs():
    """Drain any queued logs into session_state.raw_logs."""
    q = st.session_state.log_queue
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        st.session_state.raw_logs.append(line)

# -----------------------------------------------------------------------------
# Stable working dir per input (so resume works)
# -----------------------------------------------------------------------------
def stable_working_dir(upload_name: str, upload_bytes: bytes) -> Path:
    h = hashlib.md5(upload_bytes).hexdigest()[:12]
    base = Path(tempfile.gettempdir()) / "csvai" / f"{upload_name}-{h}"
    base.mkdir(parents=True, exist_ok=True)
    return base

# -----------------------------------------------------------------------------
# Inputs (simple)
# -----------------------------------------------------------------------------
with st.form("csvai_form", clear_on_submit=False):
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    prompt_src = st.radio("Prompt source", ["Upload file", "Paste text"], horizontal=True)
    if prompt_src == "Upload file":
        prompt_file = st.file_uploader("Prompt (.txt)", type=["txt"], key="prompt_file")
        prompt_text = None
    else:
        prompt_file = None
        prompt_text = st.text_area("Prompt text", height=160, key="prompt_text")

    schema_file = st.file_uploader("Schema (optional, .json)", type=["json"])
    model = st.text_input("Model", value=ProcessorConfig.model)
    limit = st.number_input("Row limit (0 = all new)", min_value=0, value=0, step=1)

    col1, col2 = st.columns(2)
    with col1:
        run_clicked = st.form_submit_button("▶ Run", use_container_width=True)
    with col2:
        reset_clicked = st.form_submit_button("Reset working folder", use_container_width=True)

# Reset working folder (simple: just clear state & forget output file)
if reset_clicked:
    try:
        if st.session_state.output_path:
            op = Path(st.session_state.output_path)
            if op.exists():
                op.unlink()
    except Exception:
        pass
    st.session_state.working_dir = None
    st.session_state.output_path = None
    st.session_state.raw_logs = []
    st.session_state.processor = None
    st.session_state.thread = None
    st.success("Working folder & UI state reset.")

# -----------------------------------------------------------------------------
# Start / resume run
# -----------------------------------------------------------------------------
if run_clicked:
    if not uploaded:
        st.error("Please upload an input file.")
    elif not (prompt_file or (prompt_text and prompt_text.strip())):
        st.error("Please provide a prompt file or text.")
    elif st.session_state.thread and st.session_state.thread.is_alive():
        st.error("A run is already in progress.")
    else:
        # Stable folder for this input => resume will find the same _enriched file
        workdir = Path(st.session_state.working_dir) if st.session_state.working_dir else stable_working_dir(
            uploaded.name, uploaded.getvalue()
        )
        st.session_state.working_dir = str(workdir)

        # Save inputs into working dir
        input_path = workdir / uploaded.name
        input_path.write_bytes(uploaded.getvalue())

        if prompt_file:
            prompt_path = workdir / prompt_file.name
            prompt_path.write_bytes(prompt_file.getvalue())
        else:
            prompt_path = workdir / "prompt.txt"
            prompt_path.write_text(prompt_text or "", encoding="utf-8")

        schema_path = None
        if schema_file:
            schema_path = workdir / schema_file.name
            schema_path.write_bytes(schema_file.getvalue())

        # Decide output path inside same folder so resume works
        default_out = default_output_file(input_path, None).name
        output_path = workdir / default_out
        st.session_state.output_path = str(output_path)

        # Prepare processor config
        cfg = ProcessorConfig(
            input=str(input_path),
            prompt=str(prompt_path),
            output=str(output_path),
            schema=str(schema_path) if schema_path else None,
            limit=int(limit) if limit > 0 else None,
            model=model,
        )
        processor = CSVAIProcessor(cfg)

        # Attach logging handler once
        if not st.session_state.log_handler_attached:
            handler = QueueLogHandler(st.session_state.log_queue)
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            root_logger = logging.getLogger()
            # Remove any prior queue handlers if hot-reloaded
            for h in list(root_logger.handlers):
                if isinstance(h, QueueLogHandler):
                    root_logger.removeHandler(h)
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)
            st.session_state.log_handler_attached = True

        # Clear old logs for this view
        st.session_state.raw_logs = []

        # Launch worker
        thread = threading.Thread(target=lambda: asyncio.run(processor.run()), daemon=True)
        thread.start()
        st.session_state.processor = processor
        st.session_state.thread = thread

        # tiny delay so first logs land
        time.sleep(0.1)
        st.rerun()

# -----------------------------------------------------------------------------
# Live view: raw logs + controls + download
# -----------------------------------------------------------------------------
drain_logs()
processor = st.session_state.processor
thread = st.session_state.thread

# Controls during run (optional; simple)
if thread and thread.is_alive():
    c1, c2, c3 = st.columns(3)
    paused = processor is not None and (not processor.pause_event.is_set())
    with c1:
        st.button("Pause", on_click=processor.pause, disabled=paused, use_container_width=True)
    with c2:
        st.button("Resume", on_click=processor.resume, disabled=not paused, use_container_width=True)
    with c3:
        st.button("Stop", on_click=processor.stop, use_container_width=True)

# Raw logs (read-only)
st.subheader("Logs")
if st.session_state.raw_logs:
    st.code("\n".join(st.session_state.raw_logs), language=None)
else:
    st.info("No logs yet.")

# Download (if file exists)
if st.session_state.output_path and Path(st.session_state.output_path).exists():
    outp = Path(st.session_state.output_path)
    with open(outp, "rb") as f:
        st.download_button("Download enriched file", f, file_name=outp.name, use_container_width=True)

# Auto-refresh while running to stream logs
if thread and thread.is_alive():
    time.sleep(1)
    drain_logs()
    st.rerun()

