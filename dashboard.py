import json
import os
import time
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

# =========================
# Configuration
# =========================
DEFAULT_JSON = r"G:\yoran_rl\demo_status.json"
REFRESH_SEC = 1.0
DEFAULT_TAU = 0.40

st.set_page_config(page_title="O-RAN MARL + LMUT Dashboard", layout="wide")


# =========================
# Data Loading
# =========================
def load_status(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def file_mtime_ts(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None


# =========================
# Helpers
# =========================
def safe_get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def fmt_ts(ts):
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def pick_step(meta: dict):
    """Prefer meta.step, fallback to meta.seen, else N/A."""
    if not isinstance(meta, dict):
        return "N/A"
    step = meta.get("step", None)
    if step not in [None, "", "N/A"]:
        return step
    seen = meta.get("seen", None)
    if seen not in [None, "", "N/A"]:
        return seen
    return "N/A"


def pick_last_sync(meta: dict, json_path: str):
    """Prefer meta.timestamp (writer-provided); else file mtime."""
    if isinstance(meta, dict):
        ts = meta.get("timestamp", None)
        if ts not in [None, "", "N/A"]:
            return fmt_ts(ts)
    return fmt_ts(file_mtime_ts(json_path))


# =========================
# Plot helpers
# =========================
def line_chart(y, title, color="#00CC96"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines+markers", line=dict(color=color, width=2)))
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        height=240,
        template="plotly_dark",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )
    return fig


def bar_chart(labels, values, title, color="#FFA15A"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=color))
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        template="plotly_dark",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )
    return fig


# =========================
# UI
# =========================
st.title("O-RAN MARL + Online LMUT Explainer")

with st.sidebar:
    st.header("Console Settings")
    json_path = st.text_input("Data Source Path", value=DEFAULT_JSON)
    refresh = st.checkbox("Auto Refresh", value=True)
    st.divider()
    st.caption("Monitoring demo_status.json (xApp1 + continual LMUT)")

status = load_status(json_path)

if status is None:
    st.warning(f"Waiting for data source: {json_path} ... Please start xApp1.")
else:
    meta = safe_get(status, "meta", {})
    last_sync = pick_last_sync(meta, json_path)
    step_show = pick_step(meta)

    # tau display (optional)
    tau = safe_float(meta.get("tau_cov", None), DEFAULT_TAU) if isinstance(meta, dict) else DEFAULT_TAU

    st.caption(f"Last Sync: {last_sync} | Step: {step_show}")

    # --- KPI Row ---
    kpi = safe_get(status, "kpi", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Throughput (bps)", f"{safe_float(safe_get(kpi, 'thr_bps', 0.0)):,.0f}")
    c2.metric("Avg HOL (ms)", f"{safe_float(safe_get(kpi, 'avg_hol_ms', 0.0)):.1f}")
    c3.metric("Outage Ratio", f"{safe_float(safe_get(kpi, 'outage_ratio', 0.0)):.3f}")
    c4.metric("Avg SINR (dB)", f"{safe_float(safe_get(kpi, 'avg_sinr_db', 0.0)):.2f}")
    c5.metric("Queue Ratio", f"{safe_float(safe_get(kpi, 'queue_ratio', 0.0)):.3f}")
    c6.metric("Dropped Bits", f"{safe_float(safe_get(kpi, 'drops_bits_win', 0.0)):,.0f}")

    st.divider()

    # --- Action snapshot ---
    action = safe_get(status, "action", {})
    a_scale = safe_float(safe_get(action, "A_avg_scale", 0.0))
    b_mode = safe_get(action, "B_mode", "N/A")

    a1, a2 = st.columns(2)
    a1.metric("Action A: Mean Power Scale", f"{a_scale:.3f}")
    a2.metric("Action B: Mode", f"{b_mode}")

    st.divider()

    # --- Decision trends + distributions ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Decision Trends")
        hist = safe_get(status, "history", {})
        hist_A = hist.get("A_avg_scale", [])
        if isinstance(hist_A, list) and len(hist_A) > 0:
            st.plotly_chart(line_chart(hist_A, "Action A: Mean Power Scaling (Rolling Window)"),
                            use_container_width=True)
        else:
            st.info("Waiting for Action A history...")

    with col_right:
        st.subheader("B-Mode Distribution (xApp1)")
        dist_data = safe_get(status, "dist", {}).get("B_mode_counts", {})

        if isinstance(dist_data, dict) and len(dist_data) > 0:
            try:
                keys_sorted = sorted(dist_data.keys(), key=lambda x: int(x))
            except Exception:
                keys_sorted = list(dist_data.keys())

            labels = [f"Mode {k}" for k in keys_sorted]
            values = [int(dist_data.get(k, 0)) for k in keys_sorted]

            if sum(values) == 0:
                st.info("No xApp1 distribution data collected yet.")
            else:
                st.plotly_chart(bar_chart(labels, values, "Mode Counts (Cumulative)"),
                                use_container_width=True)
        else:
            st.info("Waiting for xApp1 B-mode counts...")

        # Optional: LMUT distribution if available
        dist_lmut = safe_get(status, "dist", {}).get("B_mode_counts_lmut", {})
        if isinstance(dist_lmut, dict) and len(dist_lmut) > 0 and sum(int(v) for v in dist_lmut.values()) > 0:
            st.subheader("B-Mode Distribution (LMUT eval)")
            try:
                ks = sorted(dist_lmut.keys(), key=lambda x: int(x))
            except Exception:
                ks = list(dist_lmut.keys())
            labels2 = [f"Mode {k}" for k in ks]
            values2 = [int(dist_lmut.get(k, 0)) for k in ks]
            st.plotly_chart(bar_chart(labels2, values2, "LMUT Window Counts"),
                            use_container_width=True)

    st.divider()

    # --- Explainability Row ---
    st.subheader("LMUT Explainability")
    rel = safe_get(status, "reliability", {})
    diag = safe_get(status, "diag", {})
    rules = safe_get(status, "rules", {})

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LMUT Acc_B", f"{safe_float(safe_get(rel, 'acc_B', None)):.3f}" if rel else "N/A")
    m2.metric("Baseline_B", f"{safe_float(safe_get(rel, 'baseline_B', None)):.3f}" if rel else "N/A")
    m3.metric("MAE_A", f"{safe_float(safe_get(rel, 'mae_A', None)):.3f}" if rel else "N/A")
    m4.metric(f"Coverage @ τ={tau:.2f}", f"{safe_float(safe_get(diag, 'coverage', None)):.3f}" if diag else "N/A")
    m5.metric(f"Precision @ τ={tau:.2f}", f"{safe_float(safe_get(diag, 'precision', None)):.3f}" if diag else "N/A")
    m6.metric("Rule Stability", f"{safe_float(safe_get(diag, 'stability', None)):.3f}" if diag else "N/A")

    st.markdown("#### Interpretable Decision Rules (Top Rules)")

    rulesA = rules.get("A_rules_top", []) if isinstance(rules, dict) else []
    rulesB = rules.get("B_rules_top", []) if isinstance(rules, dict) else []

    ra, rb = st.columns(2)
    with ra:
        st.info("Action A (Power) Logic")
        if isinstance(rulesA, list) and len(rulesA) > 0:
            for r in rulesA[:3]:
                st.code(r, language="python")
        else:
            st.write("Pending / N/A")

    with rb:
        st.success("Action B (Mode) Logic")
        if isinstance(rulesB, list) and len(rulesB) > 0:
            for r in rulesB[:3]:
                st.code(r, language="python")
        else:
            st.write("Pending / N/A")

# --- Auto Refresh ---
if refresh:
    time.sleep(REFRESH_SEC)
    st.rerun()
