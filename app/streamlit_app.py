import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path("/Users/rhishabhpatil/Desktop/Portfolio/Fraud_Dashboard")
OUT = ROOT / "outputs"

st.set_page_config(page_title="Fraud Risk Monitoring", layout="wide")
st.title("Fraud Risk Monitoring Dashboard")

metrics = json.loads((OUT / "metrics" / "metrics.json").read_text())

c1, c2, c3 = st.columns(3)
c1.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
c2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
c3.metric("Best Threshold", f"{metrics['best_threshold_f1']:.3f}")

left, right = st.columns(2)

with left:
    st.subheader("Precision–Recall Curve")
    st.components.v1.html((OUT / "figures" / "pr_curve.html").read_text(), height=500)

with right:
    st.subheader("Risk Score Distribution")
    st.components.v1.html((OUT / "figures" / "score_distribution.html").read_text(), height=500)

st.subheader("Investigation Workload")
st.components.v1.html((OUT / "figures" / "workload_curve.html").read_text(), height=400)

st.subheader("Confusion Matrix")
st.write(pd.DataFrame(metrics["confusion_matrix"], columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))
