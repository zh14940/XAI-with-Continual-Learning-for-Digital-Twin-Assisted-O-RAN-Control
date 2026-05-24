# continual_lmut_task2.py
# ------------------------------------------------------------
# Continual LMUT (final): Task-2.1
#  - Predict B_mode (classification, 4 classes)
#  - Predict A_avg_scale (regression)
#
# Final fixes:
#  (1) predict_B_proba() never returns all-zeros (coverage/precision no longer stuck at 0)
#  (2) alpha has a floor (alpha_floor) to avoid all-zero ensemble weights
#  (3) rule stability (Jaccard over top rules) computed on REFRESH
#  (4) merge-write demo_status.json (do not clobber xApp1 fields)
#  (5) IMPORTANT: UPDATE no longer overwrites stability to 0 (None => keep previous)
#
# Run:
#   G:\yoran_rl\.venv\Scripts\python.exe -u G:\yoran_rl\continual_lmut_task2.py --csv G:\yoran_rl\marl_rollout_log.csv
#
# Optional:
#   --tau_cov 0.40
#   --refresh_every 2000   (to see stability change sooner)
# ------------------------------------------------------------

import os
import time
import csv
import json
import pickle
import argparse
from dataclasses import dataclass
from collections import deque, Counter
from typing import List, Optional, Tuple, Dict

import numpy as np


# ============================================================
# Dashboard status writer (merge-write, file-based)
# ============================================================

from collections import deque as _deque

_DASH_HIST_MAX = 120
_hist_accB = _deque(maxlen=_DASH_HIST_MAX)
_hist_maeA = _deque(maxlen=_DASH_HIST_MAX)

_last_rules_A: List[str] = []
_last_rules_B: List[str] = []


def _safe_float2(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int2(x, default=0):
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _atomic_write_json_retry(path: str, payload: dict, retry: int = 15):
    tmp = path + ".tmp"
    for _ in range(retry):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
        except Exception:
            time.sleep(0.05)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _jaccard(a: List[str], b: List[str]) -> float:
    A = set(a or [])
    B = set(b or [])
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / max(1, len(A | B)))


def write_demo_status_merge(
    out_path: str,
    *,
    seen: int,
    acc_B: float,
    baseline_B: float,
    mae_A: float,
    last_info: dict,
    rulesA_top: Optional[List[str]] = None,
    rulesB_top: Optional[List[str]] = None,
    B_mode_counts: Optional[Dict[int, int]] = None,
    coverage: Optional[float] = None,
    precision: Optional[float] = None,
    stability: Optional[float] = None,
    tau_cov: Optional[float] = None,
):
    """
    Merge-write into demo_status.json so we don't clobber xApp1 fields.
    Key behavior:
      - Keep existing kpi/action/history(A_avg_scale)/dist from xApp1.
      - Update: meta(timestamp, seen, step, tau_cov), reliability, rules.
      - Update diag fields ONLY if provided (None => keep previous),
        so UPDATE will NOT overwrite stability back to 0.
    """
    global _last_rules_A, _last_rules_B

    _hist_accB.append(_safe_float2(acc_B))
    _hist_maeA.append(_safe_float2(mae_A))

    if rulesA_top is not None:
        _last_rules_A = list(rulesA_top)
    if rulesB_top is not None:
        _last_rules_B = list(rulesB_top)

    existing = _read_json_safe(out_path)

    # ---- meta ----
    meta = existing.get("meta", {}) if isinstance(existing.get("meta", {}), dict) else {}
    meta["timestamp"] = time.time()
    meta["seen"] = int(seen)
    if tau_cov is not None:
        meta["tau_cov"] = float(tau_cov)

    # keep xApp1 step if exists; else take CSV iter if available
    step_from_csv = _safe_int2(last_info.get("iter", None), default=None) if isinstance(last_info, dict) else None
    if meta.get("step", None) in [None, "", "N/A"]:
        if step_from_csv is not None:
            meta["step"] = int(step_from_csv)
    existing["meta"] = meta

    # ---- reliability ----
    existing["reliability"] = {
        "acc_B": _safe_float2(acc_B),
        "baseline_B": _safe_float2(baseline_B),
        "mae_A": _safe_float2(mae_A),
    }

    # ---- diag (IMPORTANT: None => keep previous) ----
    diag_prev = existing.get("diag", {}) if isinstance(existing.get("diag", {}), dict) else {}

    if coverage is not None:
        diag_prev["coverage"] = _safe_float2(coverage, 0.0)
    else:
        diag_prev["coverage"] = _safe_float2(diag_prev.get("coverage", 0.0), 0.0)

    if precision is not None:
        diag_prev["precision"] = _safe_float2(precision, 0.0)
    else:
        diag_prev["precision"] = _safe_float2(diag_prev.get("precision", 0.0), 0.0)

    if stability is not None:
        diag_prev["stability"] = _safe_float2(stability, 0.0)
    else:
        diag_prev["stability"] = _safe_float2(diag_prev.get("stability", 0.0), 0.0)

    existing["diag"] = diag_prev

    # ---- rules ----
    existing["rules"] = {
        "A_rules_top": _last_rules_A[:5],
        "B_rules_top": _last_rules_B[:5],
    }

    # ---- dist (optional LMUT window counts under separate key) ----
    if B_mode_counts is not None:
        dist = existing.get("dist", {}) if isinstance(existing.get("dist", {}), dict) else {}
        dist["B_mode_counts_lmut"] = {str(int(k)): int(v) for k, v in B_mode_counts.items()}
        existing["dist"] = dist

    # ---- history (preserve xApp1 and add LMUT) ----
    hist = existing.get("history", {}) if isinstance(existing.get("history", {}), dict) else {}
    hist["acc_B"] = list(_hist_accB)
    hist["mae_A"] = list(_hist_maeA)
    existing["history"] = hist

    _atomic_write_json_retry(out_path, existing, retry=15)


# ============================================================
# Tree internals
# ============================================================

@dataclass
class _Node:
    is_leaf: bool
    feat: int = -1
    thr: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: float = 0.0
    proba: Optional[np.ndarray] = None


class LMUTTree:
    def __init__(
        self,
        task: str,
        n_classes: int = 4,
        max_depth: int = 3,
        min_leaf: int = 50,
        n_thresholds: int = 32,
        random_state: int = 0,
    ):
        assert task in ("classification", "regression")
        self.task = task
        self.n_classes = int(n_classes)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)
        self.root: Optional[_Node] = None
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "LMUTTree":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n = X.shape[0]
        self.n_features_ = int(X.shape[1])

        if sample_weight is None:
            w = np.ones((n,), dtype=np.float32)
        else:
            w = np.asarray(sample_weight, dtype=np.float32)
            if w.shape[0] != n:
                raise ValueError("sample_weight length mismatch")

        rng = np.random.RandomState(self.random_state)
        self.root = self._grow(X, y, w, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.task == "regression":
            out = np.zeros((X.shape[0],), dtype=np.float32)
            for i in range(X.shape[0]):
                out[i] = self._predict_one_reg(X[i], self.root)
            return out
        else:
            out = np.zeros((X.shape[0],), dtype=np.int64)
            for i in range(X.shape[0]):
                p = self._predict_one_proba(X[i], self.root)
                out[i] = int(np.argmax(p))
            return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only valid for classification")
        X = np.asarray(X, dtype=np.float32)
        P = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        for i in range(X.shape[0]):
            P[i] = self._predict_one_proba(X[i], self.root)
        return P

    def rules(self, feature_names: List[str]) -> List[str]:
        if self.root is None:
            return []
        out: List[str] = []
        self._collect_rules(self.root, conds=[], out=out, feature_names=feature_names)
        return out

    def _grow(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int, rng: np.random.RandomState) -> _Node:
        n = X.shape[0]
        if depth >= self.max_depth or n < 2 * self.min_leaf:
            return self._make_leaf(y, w)

        if self.task == "classification":
            if np.unique(y).size <= 1:
                return self._make_leaf(y, w)
        else:
            if float(np.max(y) - np.min(y)) < 1e-9:
                return self._make_leaf(y, w)

        best_loss = np.inf
        best_feat = -1
        best_thr = 0.0
        best_split = None

        feat_order = np.arange(self.n_features_, dtype=np.int32)
        rng.shuffle(feat_order)

        for f in feat_order:
            xf = X[:, f]
            thr_list = self._candidate_thresholds(xf, w, k=self.n_thresholds)
            if thr_list.size == 0:
                continue

            for thr in thr_list:
                mask = xf <= thr
                nl = int(np.sum(mask))
                nr = n - nl
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue

                yl, yr = y[mask], y[~mask]
                wl, wr = w[mask], w[~mask]
                loss = (np.sum(wl) * self._impurity(yl, wl) + np.sum(wr) * self._impurity(yr, wr)) / (np.sum(w) + 1e-12)

                if loss < best_loss:
                    best_loss = loss
                    best_feat = int(f)
                    best_thr = float(thr)
                    best_split = (mask, yl, wl, yr, wr)

        if best_feat < 0 or best_split is None:
            return self._make_leaf(y, w)

        mask, yl, wl, yr, wr = best_split
        node = _Node(is_leaf=False, feat=best_feat, thr=best_thr)
        node.left = self._grow(X[mask], yl, wl, depth + 1, rng)
        node.right = self._grow(X[~mask], yr, wr, depth + 1, rng)
        return node

    def _impurity(self, y: np.ndarray, w: np.ndarray) -> float:
        if self.task == "regression":
            wsum = float(np.sum(w))
            mu = float(np.sum(w * y) / (wsum + 1e-12))
            mse = float(np.sum(w * (y - mu) ** 2) / (wsum + 1e-12))
            return mse
        else:
            wsum = float(np.sum(w))
            if wsum <= 0:
                return 0.0
            counts = np.zeros((self.n_classes,), dtype=np.float64)
            for c in range(self.n_classes):
                counts[c] = float(np.sum(w[y == c]))
            p = counts / (wsum + 1e-12)
            return float(1.0 - np.sum(p * p))

    def _make_leaf(self, y: np.ndarray, w: np.ndarray) -> _Node:
        if self.task == "regression":
            wsum = float(np.sum(w))
            val = float(np.sum(w * y) / (wsum + 1e-12))
            return _Node(is_leaf=True, value=val, proba=None)
        else:
            wsum = float(np.sum(w))
            counts = np.zeros((self.n_classes,), dtype=np.float64)
            for c in range(self.n_classes):
                counts[c] = float(np.sum(w[y == c]))
            proba = (counts / (wsum + 1e-12)).astype(np.float32)
            return _Node(is_leaf=True, value=0.0, proba=proba)

    def _candidate_thresholds(self, x: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if float(np.max(x) - np.min(x)) < 1e-8:
            return np.array([], dtype=np.float32)

        idx = np.argsort(x)
        xs = x[idx]
        ws = w[idx].astype(np.float64)
        cdf = np.cumsum(ws)
        total = cdf[-1] if cdf.size > 0 else 1.0
        if total <= 0:
            return np.array([], dtype=np.float32)

        q = np.linspace(0.05, 0.95, num=k, dtype=np.float64)
        thr = np.interp(q * total, cdf, xs).astype(np.float32)
        return np.unique(thr)

    def _predict_one_reg(self, x: np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            node = node.left if x[node.feat] <= node.thr else node.right
        return float(node.value)

    def _predict_one_proba(self, x: np.ndarray, node: _Node) -> np.ndarray:
        while not node.is_leaf:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.proba

    def _collect_rules(self, node: _Node, conds: List[str], out: List[str], feature_names: List[str]):
        if node.is_leaf:
            if self.task == "regression":
                out.append("IF " + " AND ".join(conds) + f" THEN A_hat={node.value:.4f}")
            else:
                c = int(np.argmax(node.proba))
                p = float(np.max(node.proba))
                out.append("IF " + " AND ".join(conds) + f" THEN B_hat={c} (p={p:.3f})")
            return
        fname = feature_names[node.feat] if node.feat < len(feature_names) else f"f{node.feat}"
        self._collect_rules(node.left, conds + [f"{fname} <= {node.thr:.4f}"], out, feature_names)
        self._collect_rules(node.right, conds + [f"{fname} > {node.thr:.4f}"], out, feature_names)


# ============================================================
# CSV incremental reader
# ============================================================

def _load_csv_tail(csv_path: str, last_pos: int) -> Tuple[List[Dict[str, str]], int, Optional[List[str]]]:
    if not os.path.exists(csv_path):
        return [], last_pos, None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        f.seek(last_pos)
        chunk = f.read()
        new_pos = f.tell()

    if not chunk:
        return [], last_pos, None

    lines = chunk.splitlines()

    if chunk and not chunk.endswith("\n"):
        partial = lines[-1] if lines else ""
        new_pos -= len(partial)
        lines = lines[:-1]

    if not lines:
        return [], new_pos, None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    new_rows: List[Dict[str, str]] = []
    for ln in lines:
        if not ln.strip():
            continue
        vals = list(csv.reader([ln]))[0]
        if len(vals) != len(header):
            continue
        row = {header[i]: vals[i] for i in range(len(header))}
        new_rows.append(row)

    return new_rows, new_pos, header


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _pick_numeric_features(header: List[str], warm_rows: List[Dict[str, str]], drop: set) -> List[str]:
    candidates = [c for c in header if c not in drop]
    numeric = []
    for c in candidates:
        ok = 0
        tot = 0
        for r in warm_rows[:200]:
            tot += 1
            v = r.get(c, "")
            try:
                float(v)
                ok += 1
            except Exception:
                pass
        if tot > 0 and ok / tot >= 0.95:
            numeric.append(c)
    return numeric


def _extract_xy(
    rows: List[Dict[str, str]],
    feature_cols: List[str],
    yA_col="A_avg_scale",
    yB_col="B_mode",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.zeros((len(rows), len(feature_cols)), dtype=np.float32)
    yA = np.zeros((len(rows),), dtype=np.float32)
    yB = np.zeros((len(rows),), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, c in enumerate(feature_cols):
            X[i, j] = _safe_float(r.get(c, 0.0), 0.0)
        yA[i] = float(np.clip(_safe_float(r.get(yA_col, 0.5), 0.5), 0.0, 1.0))
        yB[i] = int(_safe_float(r.get(yB_col, 0), 0.0))
    return X, yA, yB


def _row_to_last_info(r: Dict[str, str]) -> dict:
    return {
        "avg_se": _safe_float(r.get("avg_se", 0.0), 0.0),
        "avg_sinr_db": _safe_float(r.get("avg_sinr_db", 0.0), 0.0),
        "outage_ratio": _safe_float(r.get("outage_ratio", 0.0), 0.0),
        "queue_ratio": _safe_float(r.get("queue_ratio", 0.0), 0.0),
        "avg_hol_ms": _safe_float(r.get("avg_hol_ms", 0.0), 0.0),
        "drops_bits_win": _safe_float(r.get("drops_bits_win", 0.0), 0.0),
        "A_avg_scale": float(np.clip(_safe_float(r.get("A_avg_scale", 0.5), 0.5), 0.0, 1.0)),
        "B_mode": int(_safe_float(r.get("B_mode", 0), 0.0)),
        "ts_unix": _safe_float(r.get("ts_unix", time.time()), time.time()),
        "iter": int(_safe_float(r.get("iter", 0), 0.0)),
    }


# ============================================================
# Explainability diagnostics (coverage / precision)
# ============================================================

def _coverage_precision_from_proba(P: np.ndarray, y_true: np.ndarray, tau: float) -> Tuple[float, float]:
    if P is None or P.size == 0:
        return 0.0, 0.0
    y_true = np.asarray(y_true).astype(np.int64)
    pmax = np.max(P, axis=1)
    sel = pmax >= float(tau)
    cov = float(np.mean(sel)) if sel.size > 0 else 0.0
    if np.sum(sel) == 0:
        return cov, 0.0
    y_hat = np.argmax(P, axis=1).astype(np.int64)
    prec = float(np.mean((y_hat[sel] == y_true[sel]).astype(np.float32)))
    return cov, prec


def _compute_diag_for_window(Xw: np.ndarray, yBw: np.ndarray, tau: float, proba_fn) -> Tuple[float, float]:
    try:
        P = proba_fn(Xw)
        return _coverage_precision_from_proba(P, yBw, tau)
    except Exception:
        return 0.0, 0.0


# ============================================================
# Continual runner
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min_warm", type=int, default=2000)
    ap.add_argument("--update_every", type=int, default=500)
    ap.add_argument("--refresh_every", type=int, default=20000)
    ap.add_argument("--sleep_s", type=float, default=0.5)

    ap.add_argument("--model_out", default="boosted_lmut_task2.pkl")
    ap.add_argument("--buffer_max", type=int, default=80000)

    ap.add_argument("--status_out", default=r"G:\yoran_rl\demo_status.json")
    ap.add_argument("--tau_cov", type=float, default=0.30)  # <-- FINAL: default tau = 0.30

    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_leaf", type=int, default=50)
    ap.add_argument("--n_thresholds", type=int, default=32)

    ap.add_argument("--max_trees", type=int, default=400)

    ap.add_argument("--alpha_clip", type=float, default=2.0)
    ap.add_argument("--alpha_lr", type=float, default=1.0)
    ap.add_argument("--alpha_floor", type=float, default=0.05)

    ap.add_argument("--beta_err", type=float, default=4.0)
    ap.add_argument("--gamma_queue", type=float, default=1.5)
    ap.add_argument("--gamma_deadline", type=float, default=1.2)
    ap.add_argument("--gamma_outage", type=float, default=1.0)
    ap.add_argument("--gamma_lowsinr", type=float, default=1.0)
    ap.add_argument("--sinr_ref_db", type=float, default=5.0)

    ap.add_argument("--gammaA_queue", type=float, default=0.5)

    args = ap.parse_args()

    print(f"[LMUT] Watching CSV: {args.csv}", flush=True)
    print(f"[LMUT] update_every={args.update_every}, refresh_every={args.refresh_every}, min_warm={args.min_warm}", flush=True)
    print(f"[LMUT] tau_cov={args.tau_cov}, alpha_floor={args.alpha_floor}", flush=True)

    buf: deque = deque(maxlen=args.buffer_max)
    last_pos = 0

    seen = 0
    last_update_seen = 0
    last_refresh_seen = 0

    feature_cols: Optional[List[str]] = None
    feat_idx: Dict[str, int] = {}

    treesA: List[LMUTTree] = []
    treesB: List[LMUTTree] = []
    alphasB: List[float] = []

    def predict_A(X: np.ndarray) -> np.ndarray:
        if not treesA:
            return np.full((X.shape[0],), 0.5, dtype=np.float32)
        ps = np.stack([t.predict(X) for t in treesA], axis=0).astype(np.float32)
        return np.mean(ps, axis=0)

    def predict_B_proba(X: np.ndarray) -> np.ndarray:
        if not treesB:
            P = np.zeros((X.shape[0], 4), dtype=np.float32)
            P[:, 0] = 1.0
            return P

        a_sum = float(np.sum(alphasB)) if alphasB else 0.0
        if a_sum <= 1e-12:
            Ps = [t.predict_proba(X) for t in treesB]
            P = np.mean(np.stack(Ps, axis=0).astype(np.float32), axis=0)
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
            return P

        P = np.zeros((X.shape[0], 4), dtype=np.float32)
        for t, a in zip(treesB, alphasB):
            P += float(a) * t.predict_proba(X)
        P /= max(a_sum, 1e-12)
        P = P / (np.sum(P, axis=1, keepdims=True) + 1e-12)
        return P

    def predict_B(X: np.ndarray) -> np.ndarray:
        return np.argmax(predict_B_proba(X), axis=1).astype(np.int64)

    def _get_col(X: np.ndarray, name: str) -> Optional[np.ndarray]:
        if name in feat_idx:
            return X[:, feat_idx[name]]
        return None

    def _hard_factor_B(X: np.ndarray) -> np.ndarray:
        f = np.ones((X.shape[0],), dtype=np.float32)

        q = _get_col(X, "queue_ratio")
        if q is not None:
            f *= (1.0 + args.gamma_queue * np.clip(q, 0.0, 1.0)).astype(np.float32)

        out = _get_col(X, "outage_ratio")
        if out is not None:
            f *= (1.0 + args.gamma_outage * np.clip(out, 0.0, 1.0)).astype(np.float32)

        sinr = _get_col(X, "avg_sinr_db")
        if sinr is not None:
            low = np.clip((args.sinr_ref_db - sinr) / max(args.sinr_ref_db, 1e-6), 0.0, 2.0)
            f *= (1.0 + args.gamma_lowsinr * low).astype(np.float32)

        return np.clip(f, 1.0, 10.0).astype(np.float32)

    def make_weights_for_B(X: np.ndarray, yB: np.ndarray) -> np.ndarray:
        counts = np.bincount(yB, minlength=4).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        inv = inv / np.mean(inv)
        w = inv[yB].astype(np.float32)

        if treesB:
            pred = predict_B(X)
            w *= (1.0 + args.beta_err * (pred != yB).astype(np.float32))

        w *= _hard_factor_B(X)
        return np.clip(w, 1e-3, 20.0).astype(np.float32)

    def make_weights_for_A(X: np.ndarray, yA: np.ndarray) -> np.ndarray:
        if not treesA:
            w = np.ones((len(yA),), dtype=np.float32)
        else:
            pred = predict_A(X)
            err = np.abs(pred - yA).astype(np.float32)
            w = 1.0 + 2.0 * (err / (float(np.mean(err)) + 1e-6))
            w = np.clip(w, 1.0, 5.0).astype(np.float32)

        q = _get_col(X, "queue_ratio")
        if q is not None and args.gammaA_queue > 0:
            w *= (1.0 + args.gammaA_queue * np.clip(q, 0.0, 1.0)).astype(np.float32)

        return np.clip(w, 1e-3, 20.0).astype(np.float32)

    warm_probe_rows: List[Dict[str, str]] = []

    # initial status
    try:
        write_demo_status_merge(
            args.status_out,
            seen=0,
            acc_B=0.0,
            baseline_B=0.0,
            mae_A=0.0,
            last_info={},
            rulesA_top=[],
            rulesB_top=[],
            B_mode_counts={},
            coverage=0.0,
            precision=0.0,
            stability=0.0,
            tau_cov=args.tau_cov,
        )
    except Exception:
        pass

    prev_rulesA_top: List[str] = []
    prev_rulesB_top: List[str] = []

    while True:
        new_rows, last_pos2, header = _load_csv_tail(args.csv, last_pos)

        if new_rows:
            last_pos = last_pos2
            for r in new_rows:
                buf.append(r)
                warm_probe_rows.append(r)
                seen += 1

        if feature_cols is None and header is not None and len(warm_probe_rows) >= min(args.min_warm, 500):
            drop = set(["B_mode", "A_avg_scale"])
            feature_cols = _pick_numeric_features(header, warm_probe_rows, drop=drop)
            feat_idx = {c: i for i, c in enumerate(feature_cols)}
            print(f"[LMUT] feature_cols={len(feature_cols)} -> {feature_cols}", flush=True)

        if feature_cols is None or seen < args.min_warm:
            time.sleep(args.sleep_s)
            continue

        last_info = _row_to_last_info(buf[-1]) if len(buf) > 0 else {}

        # ---------- UPDATE ----------
        if seen - last_update_seen >= args.update_every:
            last_update_seen = seen
            rows = list(buf)

            train_win = min(30000, len(rows))
            eval_win = min(5000, len(rows))

            Xtr, yAtr, yBtr = _extract_xy(rows[-train_win:], feature_cols)
            Xv, yAv, yBv = _extract_xy(rows[-eval_win:], feature_cols)

            wB = make_weights_for_B(Xtr, yBtr)
            tB = LMUTTree(
                task="classification",
                n_classes=4,
                max_depth=args.max_depth,
                min_leaf=args.min_leaf,
                n_thresholds=args.n_thresholds,
                random_state=seen,
            ).fit(Xtr, yBtr, sample_weight=wB)

            predBv_tree = tB.predict(Xv)
            acc_tree = float(np.mean((predBv_tree == yBv).astype(np.float32)))

            cnt_eval = np.bincount(yBv, minlength=4).astype(np.float32)
            base = float(np.max(cnt_eval) / (np.sum(cnt_eval) + 1e-12))

            margin = acc_tree - base
            alpha = args.alpha_lr * (margin / max(1e-6, (1.0 - base)))
            alpha = float(np.clip(alpha, 0.0, args.alpha_clip))
            alpha = float(max(alpha, args.alpha_floor))

            treesB.append(tB)
            alphasB.append(alpha)

            wA = make_weights_for_A(Xtr, yAtr)
            tA = LMUTTree(
                task="regression",
                max_depth=args.max_depth,
                min_leaf=args.min_leaf,
                n_thresholds=args.n_thresholds,
                random_state=seen + 7,
            ).fit(Xtr, yAtr, sample_weight=wA)
            treesA.append(tA)

            if len(treesB) > args.max_trees:
                treesB.pop(0)
                alphasB.pop(0)
            if len(treesA) > args.max_trees:
                treesA.pop(0)

            accB = _acc(yBv, predict_B(Xv))
            maeA = _mae(yAv, predict_A(Xv))

            B_counts = dict(Counter(yBv.tolist()))

            tau = float(args.tau_cov)
            cov_tau, prec_tau = _compute_diag_for_window(Xv, yBv, tau, predict_B_proba)

            print(
                f"[LMUT][UPDATE] seen={seen} acc_B={accB:.3f} mae_A={maeA:.3f} "
                f"treesA={len(treesA)} treesB={len(treesB)} "
                f"(tree_acc={acc_tree:.3f} base={base:.3f}) cov@{tau:.2f}={cov_tau:.3f} prec@{tau:.2f}={prec_tau:.3f}",
                flush=True,
            )

            # IMPORTANT: stability=None, writer will KEEP previous stability (no overwrite to 0)
            try:
                write_demo_status_merge(
                    args.status_out,
                    seen=seen,
                    acc_B=accB,
                    baseline_B=base,
                    mae_A=maeA,
                    last_info=last_info,
                    rulesA_top=None,
                    rulesB_top=None,
                    B_mode_counts=B_counts,
                    coverage=cov_tau,
                    precision=prec_tau,
                    stability=None,
                    tau_cov=args.tau_cov,
                )
            except Exception:
                pass

        # ---------- REFRESH / SAVE ----------
        if seen - last_refresh_seen >= args.refresh_every:
            last_refresh_seen = seen
            rows = list(buf)

            X, yA, yB = _extract_xy(rows, feature_cols)
            accB_all = _acc(yB, predict_B(X))
            maeA_all = _mae(yA, predict_A(X))

            payload = {
                "feature_cols": feature_cols,
                "feat_idx": feat_idx,
                "treesA": treesA,
                "treesB": treesB,
                "alphasB": alphasB,
                "seen": seen,
                "accB": accB_all,
                "maeA": maeA_all,
                "hyper": vars(args),
            }
            tmp = args.model_out + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp, args.model_out)

            cnt_all = Counter(yB.tolist())
            total = sum(cnt_all.values()) + 1e-12
            ratio = {k: round(v / total, 3) for k, v in cnt_all.items()}

            print(
                f"[LMUT][REFRESH] seen={seen} acc_B={accB_all:.3f} mae_A={maeA_all:.3f} "
                f"treesA={len(treesA)} treesB={len(treesB)} saved={args.model_out} B_ratio={ratio}",
                flush=True,
            )

            try:
                rulesA = (treesA[-1].rules(feature_cols) if treesA else [])
                rulesB = (treesB[-1].rules(feature_cols) if treesB else [])

                rulesA_top = rulesA[:5]
                rulesB_top = rulesB[:5]

                stabA = _jaccard(prev_rulesA_top, rulesA_top)
                stabB = _jaccard(prev_rulesB_top, rulesB_top)
                stability = 0.5 * (stabA + stabB)

                prev_rulesA_top = list(rulesA_top)
                prev_rulesB_top = list(rulesB_top)

                cnt_full = np.bincount(yB, minlength=4).astype(np.float32)
                base_full = float(np.max(cnt_full) / (np.sum(cnt_full) + 1e-12))

                tau = float(args.tau_cov)
                diag_win = min(20000, len(rows))
                Xd, _, yBd = _extract_xy(rows[-diag_win:], feature_cols)
                cov_tau, prec_tau = _compute_diag_for_window(Xd, yBd, tau, predict_B_proba)

                write_demo_status_merge(
                    args.status_out,
                    seen=seen,
                    acc_B=accB_all,
                    baseline_B=base_full,
                    mae_A=maeA_all,
                    last_info=last_info,
                    rulesA_top=rulesA_top,
                    rulesB_top=rulesB_top,
                    B_mode_counts=dict(cnt_all),
                    coverage=cov_tau,
                    precision=prec_tau,
                    stability=stability,
                    tau_cov=args.tau_cov,
                )
            except Exception:
                pass

        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
