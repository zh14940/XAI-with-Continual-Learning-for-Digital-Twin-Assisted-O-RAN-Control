"""
tree_viz.py  --  LMUT decision-tree activation-path visualiser (static)
-----------------------------------------------------------------------
Pick a step (or unix time), load the boosted LMUT model, feed that step's
state vector through the tree, and render the tree with the ACTIVATED path
highlighted (root -> ... -> leaf).

Two ways to use it:

  (A) Standalone Streamlit page:
        streamlit run tree_viz.py
      Talks to the query_server (default http://127.0.0.1:8765) to fetch the
      event at a given step/time, then renders.

  (B) Imported into your existing query_ui.py as an extra tab:
        import tree_viz
        tree_viz.render_tab(api_base="http://127.0.0.1:8765")
      (See the snippet at the bottom of this file for the 3 lines to add.)

Requirements:
  pip install graphviz streamlit requests numpy
  AND the Graphviz system binaries (the `dot` executable):
     Windows: winget install graphviz   (or download from graphviz.org)
              then ensure dot.exe is on PATH (restart the shell after install)

The model pickle (default boosted_lmut_task2.pkl) stores LMUTTree/_Node
objects, so this module imports the LMUT source module to make those classes
importable for unpickling.
"""

import os
import sys
import glob
import json
import pickle
import importlib.util
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import requests
import streamlit as st


# ----------------------------------------------------------------------
# 1. Make LMUTTree / _Node importable so pickle.load can resolve them.
#    The pickle was written by continual_lmut_task2.py, so the classes
#    live in that module's namespace.
# ----------------------------------------------------------------------

def _ensure_lmut_classes(lmut_module_path: Optional[str] = None):
    """
    Import the continual LMUT module under its ORIGINAL module name so that
    pickled references like `continual_lmut_task2.LMUTTree` resolve.
    Returns the imported module.
    """
    candidates = []
    if lmut_module_path:
        candidates.append(lmut_module_path)
    # Common names on disk
    candidates += [
        "continual_lmut_task2.py",
        "continual_lmut_task2__1_.py",
    ]
    # Anything matching the pattern in cwd
    candidates += glob.glob("continual_lmut_task2*.py")

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        # The pickle stores the module name as whatever __name__ was at pickle
        # time. continual_lmut_task2.py is run as "__main__" when executed
        # directly, but pickled class refs use the class's __module__, which
        # is the module's import name. We register it under the base name.
        mod_name = "continual_lmut_task2"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod          # so pickle can find it
        spec.loader.exec_module(mod)
        return mod, path

    raise FileNotFoundError(
        "Could not find continual_lmut_task2.py next to this script. "
        "Pickle classes (LMUTTree/_Node) can't be resolved without it."
    )


# ----------------------------------------------------------------------
# 2. Load the model pickle.
# ----------------------------------------------------------------------

def load_model(pkl_path: str, lmut_module_path: Optional[str] = None) -> Dict[str, Any]:
    _ensure_lmut_classes(lmut_module_path)
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # payload keys: feature_cols, feat_idx, treesA, treesB, alphasB, seen, ...
    return payload


# ----------------------------------------------------------------------
# 3. Build the state vector X for a given event, in feature_cols order.
#    events.jsonl fields are "kpi.avg_se", "action.A_avg_scale", etc.
#    feature_cols are the RAW csv column names ("avg_se", "queue_ratio", ...).
#    We map by stripping the "kpi." / "action." prefixes.
# ----------------------------------------------------------------------

def event_to_vector(event: Dict[str, Any], feature_cols: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return (X row of shape [1, n_feat], dict feature_name->value)."""
    # Build a flat lookup of raw-name -> value from the event.
    flat: Dict[str, float] = {}
    for k, v in event.items():
        if not isinstance(v, (int, float)):
            continue
        raw = k
        for pref in ("kpi.", "action.", "lmut."):
            if raw.startswith(pref):
                raw = raw[len(pref):]
                break
        flat[raw] = float(v)

    x = np.zeros((1, len(feature_cols)), dtype=np.float32)
    used: Dict[str, float] = {}
    for j, c in enumerate(feature_cols):
        val = flat.get(c, 0.0)
        x[0, j] = val
        used[c] = val
    return x, used


# ----------------------------------------------------------------------
# 4. Trace the activation path through ONE LMUTTree for a single sample.
#    Mirrors LMUTTree._predict_one_* : go left if x[feat] <= thr else right.
# ----------------------------------------------------------------------

def trace_path(tree, x_row: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Walk the tree for x_row (1-D vector). Return node list with an 'active'
    flag and the decision taken, plus the final leaf summary.
    Node ids are stable within one tree (DFS index).
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    counter = {"i": 0}

    def visit(node, parent_id, branch_label) -> int:
        nid = counter["i"]
        counter["i"] += 1
        is_leaf = bool(getattr(node, "is_leaf", False))
        entry = {
            "id": nid,
            "is_leaf": is_leaf,
            "feat": int(getattr(node, "feat", -1)),
            "thr": float(getattr(node, "thr", 0.0)),
            "value": float(getattr(node, "value", 0.0)),
            "active": False,
        }
        if is_leaf:
            proba = getattr(node, "proba", None)
            if proba is not None:
                arr = np.asarray(proba, dtype=float).ravel()
                entry["leaf_class"] = int(np.argmax(arr))
                entry["leaf_p"] = float(np.max(arr))
                entry["proba"] = arr.tolist()
            else:
                entry["leaf_value"] = float(getattr(node, "value", 0.0))
        else:
            fi = entry["feat"]
            entry["feat_name"] = feature_cols[fi] if 0 <= fi < len(feature_cols) else f"f{fi}"
        nodes.append(entry)
        if parent_id is not None:
            edges.append({"src": parent_id, "dst": nid, "label": branch_label})

        if not is_leaf:
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if left is not None:
                visit(left, nid, "<=")
            if right is not None:
                visit(right, nid, ">")
        return nid

    if getattr(tree, "root", None) is None:
        return {"nodes": [], "edges": [], "active_ids": [], "leaf": None}

    visit(tree.root, None, "")

    # Now walk again following the data to mark the active path.
    active_ids: List[int] = []
    # Re-traverse using the same DFS order to map node objects to ids.
    # Simpler: redo the live walk and record decisions, matching by structure.
    id_iter = {"i": 0}
    leaf_info = {}

    def live_walk(node):
        nid = id_iter["i"]
        id_iter["i"] += 1
        active_ids.append(nid)
        nodes[nid]["active"] = True
        if getattr(node, "is_leaf", False):
            leaf_info.update(nodes[nid])
            return
        fi = int(node.feat)
        xv = float(x_row[fi]) if 0 <= fi < x_row.shape[0] else 0.0
        go_left = xv <= float(node.thr)
        # We must advance id_iter past the subtree we DON'T take, to keep ids aligned.
        # Easiest: compute subtree sizes.
        def subtree_size(n):
            if getattr(n, "is_leaf", False):
                return 1
            s = 1
            if getattr(n, "left", None) is not None:
                s += subtree_size(n.left)
            if getattr(n, "right", None) is not None:
                s += subtree_size(n.right)
            return s

        nodes[nid]["chosen"] = "<=" if go_left else ">"
        nodes[nid]["x_value"] = xv
        if go_left:
            live_walk(node.left)
            # skip right subtree ids
            id_iter["i"] += subtree_size(node.right) if getattr(node, "right", None) else 0
        else:
            # skip left subtree ids first
            id_iter["i"] += subtree_size(node.left) if getattr(node, "left", None) else 0
            live_walk(node.right)

    live_walk(tree.root)

    return {"nodes": nodes, "edges": edges, "active_ids": active_ids, "leaf": leaf_info}


# ----------------------------------------------------------------------
# 5. Render to Graphviz DOT.
# ----------------------------------------------------------------------

_ACTIVE_FILL = "#2C6FB5"
_ACTIVE_FONT = "#FFFFFF"
_INACTIVE_FILL = "#EEF2F6"
_INACTIVE_FONT = "#586069"
_ACTIVE_EDGE = "#2C6FB5"
_INACTIVE_EDGE = "#C8D0D8"


def to_dot(trace: Dict[str, Any], title: str = "") -> str:
    lines = ["digraph LMUT {", 'rankdir=LR;', 'bgcolor="transparent";',
             'node [shape=circle, style="filled", fontname="Helvetica", fontsize=10, width=0.5];',
             'edge [fontname="Helvetica", fontsize=9];']
    if title:
        lines.append(f'labelloc="t"; label="{title}"; fontname="Helvetica"; fontsize=12;')

    for n in trace["nodes"]:
        nid = n["id"]
        active = n["active"]
        fill = _ACTIVE_FILL if active else _INACTIVE_FILL
        font = _ACTIVE_FONT if active else _INACTIVE_FONT
        pen = "2.5" if active else "1.0"
        if n["is_leaf"]:
            if "leaf_class" in n:
                lbl = f"Leaf\\nB={n['leaf_class']}\\np={n['leaf_p']:.2f}"
            else:
                lbl = f"Leaf\\nA={n.get('leaf_value', n.get('value', 0.0)):.3f}"
            shape = "doublecircle" if active else "circle"
            lines.append(
                f'n{nid} [label="{lbl}", fillcolor="{fill}", fontcolor="{font}", '
                f'color="{_ACTIVE_EDGE if active else _INACTIVE_EDGE}", penwidth={pen}, shape={shape}, width=0.7];'
            )
        else:
            fname = n.get("feat_name", f"f{n['feat']}")
            if active and "x_value" in n:
                lbl = f"{fname}\\n{n['x_value']:.3g} {n.get('chosen','')} {n['thr']:.3g}"
            else:
                lbl = f"{fname}\\n<= {n['thr']:.3g}"
            lines.append(
                f'n{nid} [label="{lbl}", fillcolor="{fill}", fontcolor="{font}", '
                f'color="{_ACTIVE_EDGE if active else _INACTIVE_EDGE}", penwidth={pen}, width=0.9];'
            )

    active_set = set(trace["active_ids"])
    for e in trace["edges"]:
        src, dst = e["src"], e["dst"]
        on_path = src in active_set and dst in active_set
        # Edge is part of activation only if both endpoints active AND it is the chosen branch
        chosen = trace["nodes"][src].get("chosen")
        is_chosen_edge = on_path and (
            (chosen == "<=" and e["label"] == "<=") or (chosen == ">" and e["label"] == ">")
        )
        col = _ACTIVE_EDGE if is_chosen_edge else _INACTIVE_EDGE
        pen = "2.5" if is_chosen_edge else "1.0"
        lines.append(f'n{src} -> n{dst} [label="{e["label"]}", color="{col}", penwidth={pen}, fontcolor="{col}"];')

    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 6. Fetch an event from the query_server by step or time.
# ----------------------------------------------------------------------

def fetch_event_by_time(api_base: str, t: float) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{api_base}/jump_to_time", params={"t": t}, timeout=10)
        j = r.json()
        if j.get("ok"):
            return j["result"]["event"]
    except Exception:
        pass
    return None


def fetch_run_summary(api_base: str) -> Dict[str, Any]:
    try:
        return requests.get(f"{api_base}/run_summary", timeout=10).json()
    except Exception:
        return {"ok": False}


# ----------------------------------------------------------------------
# 7. Select which tree to show.
#    treesB / treesA are ENSEMBLES. For a single activation-path view we show
#    the most influential single tree:
#      - classification (B): the tree with the largest alpha (most weight)
#      - regression (A): the most recently added tree (treesA[-1])
#    plus we report the full-ensemble prediction so the leaf is in context.
# ----------------------------------------------------------------------

def pick_treeB(model: Dict[str, Any]) -> Tuple[Optional[Any], int, float]:
    treesB = model.get("treesB", []) or []
    alphasB = model.get("alphasB", []) or []
    if not treesB:
        return None, -1, 0.0
    if alphasB and len(alphasB) == len(treesB):
        idx = int(np.argmax(alphasB))
        return treesB[idx], idx, float(alphasB[idx])
    return treesB[-1], len(treesB) - 1, 0.0


def pick_treeA(model: Dict[str, Any]) -> Tuple[Optional[Any], int]:
    treesA = model.get("treesA", []) or []
    if not treesA:
        return None, -1
    return treesA[-1], len(treesA) - 1


def ensemble_predict(model: Dict[str, Any], x: np.ndarray) -> Dict[str, Any]:
    """Full-ensemble prediction for context (B class + proba, A value)."""
    out: Dict[str, Any] = {}
    treesB = model.get("treesB", []) or []
    alphasB = model.get("alphasB", []) or []
    treesA = model.get("treesA", []) or []

    if treesB:
        a_sum = float(np.sum(alphasB)) if alphasB else 0.0
        P = np.zeros((1, getattr(treesB[0], "n_classes", 4)), dtype=np.float32)
        if a_sum <= 1e-12:
            Ps = [t.predict_proba(x) for t in treesB]
            P = np.mean(np.stack(Ps, axis=0).astype(np.float32), axis=0)
        else:
            for t, a in zip(treesB, alphasB):
                P += float(a) * t.predict_proba(x)
            P /= max(a_sum, 1e-12)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        out["B_proba"] = P.ravel().tolist()
        out["B_pred"] = int(np.argmax(P, axis=1)[0])

    if treesA:
        ps = np.stack([t.predict(x) for t in treesA], axis=0).astype(np.float32)
        out["A_pred"] = float(np.mean(ps, axis=0)[0])

    return out


# ----------------------------------------------------------------------
# 8. The reusable tab renderer (import this into query_ui.py).
# ----------------------------------------------------------------------

def render_tab(api_base: str = "http://127.0.0.1:8765",
               default_pkl: str = "boosted_lmut_task2.pkl"):
    st.subheader("LMUT Activation Path (static)")

    cset = st.columns([2, 1, 1])
    pkl_path = cset[0].text_input("Model pickle", value=default_pkl, key="tv_pkl")
    task = cset[1].selectbox("Task", ["B (mode / classification)", "A (power / regression)"], key="tv_task")
    summary = fetch_run_summary(api_base)
    last_t = None
    if summary.get("ok"):
        last_t = summary["result"].get("last_t")
    t_default = float(last_t) if last_t else 0.0
    t_in = cset[2].number_input("Unix time t", value=t_default, format="%.3f", key="tv_t")

    if st.button("Render activation path", key="tv_go"):
        # 1. event
        event = fetch_event_by_time(api_base, t_in)
        if event is None:
            st.error("Could not fetch an event at that time from the query server.")
            return

        # 2. model
        if not os.path.exists(pkl_path):
            st.error(f"Model pickle not found: {pkl_path}")
            return
        try:
            model = load_model(pkl_path)
        except Exception as e:
            st.error(f"Failed to load model: {type(e).__name__}: {e}")
            return

        feature_cols = model.get("feature_cols") or []
        if not feature_cols:
            st.error("Model has no feature_cols; was it saved before warm-up completed?")
            return

        # 3. vector
        x, used = event_to_vector(event, feature_cols)

        # 4. pick tree + trace
        if task.startswith("B"):
            tree, idx, alpha = pick_treeB(model)
            tree_label = f"treesB[{idx}]  (alpha={alpha:.3f}, most-weighted)"
        else:
            tree, idx = pick_treeA(model)
            alpha = None
            tree_label = f"treesA[{idx}]  (latest)"

        if tree is None:
            st.warning("No tree of that type in the model yet.")
            return

        trace = trace_path(tree, x[0], feature_cols)

        # 5. ensemble context
        ctx = ensemble_predict(model, x)

        # ---- layout ----
        left, right = st.columns([3, 2])
        with left:
            dot = to_dot(trace, title=tree_label)
            st.graphviz_chart(dot, use_container_width=True)
        with right:
            st.markdown(f"**Event** — step `{event.get('step')}`, t=`{event.get('t')}`")
            # show the features actually fed in
            st.markdown("**State vector (fed to tree):**")
            st.json({k: round(v, 4) for k, v in used.items()})
            st.markdown("**Ensemble prediction (all trees):**")
            st.json(ctx)
            leaf = trace.get("leaf") or {}
            if leaf:
                if "leaf_class" in leaf:
                    st.success(f"This tree's leaf → B={leaf['leaf_class']} (p={leaf.get('leaf_p', 0):.3f})")
                elif "leaf_value" in leaf or "value" in leaf:
                    st.success(f"This tree's leaf → A_hat={leaf.get('leaf_value', leaf.get('value', 0)):.4f}")

        # 6. also show the human-readable rule of the activated leaf
        st.markdown("**Activated rule (this tree):**")
        conds = []
        for nid in trace["active_ids"]:
            n = trace["nodes"][nid]
            if not n["is_leaf"] and "chosen" in n:
                op = n["chosen"]
                conds.append(f"{n.get('feat_name','f'+str(n['feat']))} {op} {n['thr']:.4f}")
        rule_txt = "IF " + " AND ".join(conds) if conds else "(root is a leaf)"
        if trace.get("leaf"):
            lf = trace["leaf"]
            if "leaf_class" in lf:
                rule_txt += f"  THEN B={lf['leaf_class']} (p={lf.get('leaf_p',0):.3f})"
            else:
                rule_txt += f"  THEN A_hat={lf.get('leaf_value', lf.get('value',0)):.4f}"
        st.code(rule_txt, language="text")


# ----------------------------------------------------------------------
# 9. Standalone mode.
# ----------------------------------------------------------------------

def _standalone():
    st.set_page_config(page_title="LMUT Activation Path", layout="wide")
    st.title("LMUT Decision-Tree Activation Path")
    api = st.text_input("Query server base URL", value="http://127.0.0.1:8765")
    render_tab(api_base=api)


if __name__ == "__main__":
    _standalone()


# ======================================================================
# To add this as a tab inside your existing query_ui.py, add near the top:
#
#     import tree_viz
#
# and at the very bottom of query_ui.py (after the LLM chat section):
#
#     st.divider()
#     tree_viz.render_tab(api_base=API)
#
# (API is the constant already defined at the top of query_ui.py.)
# ======================================================================
