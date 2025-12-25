# gui_app.py — FINAL VERSION WITH CSP SUPPORT
"""
Streamlit GUI for EEG Pipelines:
- advanced_hybrid_pipeline.py
- hybrid_pipeline.py
- csp_pipeline.py
Includes:
  - Subject selection
  - Pipeline execution
  - Logs streamed in GUI
  - Output file browser + preview
  - Download buttons
  - Research paper link
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
from typing import List
import time
import pandas as pd

# Paths
OUTPUT_DIR = Path("outputs")
ADVANCED_SCRIPT = Path("advanced_hybrid_pipeline.py")
HYBRID_SCRIPT  = Path("hybrid_pipeline.py")
CSP_SCRIPT     = Path("pipeline_paper.py")

PAPER_LOCAL_PATH = "/mnt/data/5621-5923-1-PB.pdf"

# ----- Streamlit settings -----
st.set_page_config(page_title="EEG Hybrid/CSP GUI", layout="wide")
st.title("🧠 EEG Classification – Hybrid / CSP / Advanced GUI")

st.markdown("""
This GUI runs your ML pipelines:

📌 **Advanced Hybrid** (DWT + FBCSP + Riemannian)  
📌 **Hybrid** (DWT + CSP)  
📌 **CSP-only** pipeline  

All outputs (CSV, PNG) will appear inside the `outputs/` folder.
""")

# ----- Sidebar -----
st.sidebar.header("Pipeline Controls")

subject_input = st.sidebar.text_input(
    "Enter subjects (example: `1 2 3 4` or `1-10`):",
    value="1 2 3 4 5"
)

pipeline_choice = st.sidebar.selectbox(
    "Choose pipeline",
    ("advanced_hybrid_pipeline", "hybrid_pipeline", "pipeline_paper")
)

use_smote = st.sidebar.checkbox("Apply SMOTE (if supported)", value=False)
force_resample = st.sidebar.checkbox("Force resample to 160 Hz", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("📄 **Research Paper**")
st.sidebar.markdown(f"[Open PDF]({PAPER_LOCAL_PATH})")

# ----- Helper -----
def parse_subjects(text: str):
    text = text.replace(",", " ").replace("..", " ").strip()
    parts = text.split()
    subs = []

    for p in parts:
        if p.isdigit():
            subs.append(int(p))
        elif "-" in p:
            try:
                a, b = p.split("-", 1)
                a = int(a); b = int(b)
                subs.extend(list(range(a, b+1)))
            except:
                pass

    # remove duplicates
    return sorted(list(dict.fromkeys(subs)))


subjects = parse_subjects(subject_input)
st.sidebar.write(f"Parsed subjects: {subjects}")

# ----- Runner helpers -----
def run_subprocess(script_path: Path, subjects: list):
    py = sys.executable
    subj_args = [str(s) for s in subjects]
    cmd = [py, str(script_path), "--subjects"] + subj_args

    st.info("Running command: " + " ".join(cmd))

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    log_box = st.empty()
    for line in process.stdout:
        log_box.text(line)

    process.wait()
    return process.returncode


def run_pipeline(script_name: str, subjects: list):
    try:
        module = __import__(script_name)
        if hasattr(module, "main"):
            st.success(f"Running {script_name}.main({subjects}) directly")
            module.main(subjects)
            return 0
    except Exception as e:
        st.warning(f"Direct import failed ({e}). Falling back to subprocess.")

    script_file = Path(script_name + ".py")
    if not script_file.exists():
        st.error(f"Script `{script_file}` not found.")
        return 1

    return run_subprocess(script_file, subjects)


# ----- Buttons -----
run_button = st.button("▶ Run Pipeline")
refresh_button = st.button("🔄 Refresh Outputs")

# ----- Output browser -----
st.subheader("📁 Output Files")

col1, col2 = st.columns([2, 1])

with col2:
    if OUTPUT_DIR.exists():
        files = sorted(OUTPUT_DIR.iterdir(), reverse=True)
        for f in files:
            st.write(f"📄 {f.name}")
            if f.suffix.lower() in [".csv", ".png", ".jpg", ".jpeg"]:
                st.download_button(
                    label=f"⬇ Download {f.name}",
                    data=open(f, "rb").read(),
                    file_name=f.name
                )
    else:
        st.info("No outputs yet. Run a pipeline.")

# ----- Log Panel -----
with col1:
    st.subheader("Console Output / Logs")
    log_area = st.empty()


# ----- Pipeline Execution -----
if run_button:
    if not subjects:
        st.error("❗ No subjects detected. Please enter valid subject numbers.")
    else:
        st.success(f"Running `{pipeline_choice}` for subjects {subjects}...")
        start = time.time()

        exit_code = run_pipeline(pipeline_choice, subjects)

        elapsed = time.time() - start
        if exit_code == 0:
            st.success(f"✔ Pipeline finished in {elapsed:.1f} seconds.")
        else:
            st.error(f"❌ Pipeline exited with code {exit_code}.")

# Refresh
if refresh_button:
    st.experimental_rerun()

# ----- Quick previews (Images & CSV) -----
st.markdown("---")
st.subheader("📊 Preview of Key Outputs")

preview_targets = [
    "Accuracy_Comparison.png",
    "accuracy_comparison.png",
    "CM_LDA.png",
    "CM_SVM.png",
    "CM_MLP.png",
    "CM_MDM.png",
    "CSP_feature_means.png",
    "all_features_advanced.csv",
    "all_DWT_CSP_features.csv",
    "all_subject_csp_features.csv",
]

for file in preview_targets:
    p = OUTPUT_DIR / file
    if p.exists():
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            st.image(str(p), caption=p.name)
        elif p.suffix.lower() == ".csv":
            st.write(f"📄 {p.name}")
            df = pd.read_csv(p)
            st.dataframe(df.head(20))

st.markdown("---")
st.caption("EEG GUI – supports Hybrid, Advanced Hybrid, and CSP pipelines.")
