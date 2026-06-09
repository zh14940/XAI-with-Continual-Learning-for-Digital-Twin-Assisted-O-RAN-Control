"""
query_tools.py  --  D2: deterministic query primitives
------------------------------------------------------
Pure functions over runs/<run_id>/events.jsonl.

All tools:
  * take JSON-serialisable arguments,
  * return JSON-serialisable dicts,
  * are deterministic and side-effect-free.

These are exactly what the LLM is allowed to call. Nothing else.
"""

import os
import json
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# ----------------------------- loader -----------------------------

class EventStore:
    """In-memory event store with cheap lookups. Re-loads on file growth."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self._events: List[dict] = []
        self._last_size = 0
        self._step_index: Dict[int, int] = {}   # step -> idx
        self._t_index: List[Tuple[float, int]] = []  # sorted by t

    def manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def reload_if_changed(self) -> None:
        if not self.events_path.exists():
            return
        size = self.events_path.stat().st_size
        if size == self._last_size:
            return
        # Re-read incrementally
        with open(self.events_path, "r", encoding="utf-8") as f:
            f.seek(self._last_size)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                idx = len(self._events)
                self._events.append(ev)
                if "step" in ev:
                    self._step_index[int(ev["step"])] = idx
                if "t" in ev:
                    self._t_index.append((float(ev["t"]), idx))
        self._last_size = size
        # Keep t-index sorted (events.jsonl is append-only and t is monotonic in practice)
        self._t_index.sort(key=lambda x: x[0])

    @property
    def events(self) -> List[dict]:
        return self._events

    def count(self) -> int:
        return len(self._events)

    # --- lookups ---

    def index_at_time(self, t: float) -> Optional[int]:
        """Return index of the event with the largest t <= query t (last-known-good)."""
        if not self._t_index:
            return None
        lo, hi = 0, len(self._t_index) - 1
        if t < self._t_index[0][0]:
            return None
        if t >= self._t_index[-1][0]:
            return self._t_index[-1][1]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._t_index[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        return self._t_index[lo][1]

    def index_at_step(self, step: int) -> Optional[int]:
        return self._step_index.get(int(step))

    def slice_by_time(self, t1: float, t2: float) -> List[dict]:
        if t1 > t2:
            t1, t2 = t2, t1
        out = []
        for t, idx in self._t_index:
            if t < t1:
                continue
            if t > t2:
                break
            out.append(self._events[idx])
        return out


# ----------------------------- helpers -----------------------------

_KPI_FIELDS = [
    "kpi.avg_se", "kpi.avg_hol_ms", "kpi.outage_ratio",
    "kpi.avg_sinr_db", "kpi.queue_ratio", "kpi.drops_bits_win",
]
_ACTION_FIELDS = ["action.A_avg_scale", "action.B_mode"]


def _percentiles(values: List[float], qs=(0.5, 0.9, 0.99)) -> Dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": None for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        if len(s) == 1:
            out[f"p{int(q*100)}"] = s[0]
            continue
        k = q * (len(s) - 1)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        out[f"p{int(q*100)}"] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p50": None, "p90": None, "p99": None}
    return {
        "n":    len(values),
        "mean": statistics.fmean(values),
        "std":  statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min":  min(values),
        "max":  max(values),
        **_percentiles(values),
    }


# ============================================================
# Tool 1: jump_to_time
# ============================================================

def jump_to_time(store: EventStore, t: float) -> Dict[str, Any]:
    """Return the full event snapshot at (or just before) time t."""
    store.reload_if_changed()
    idx = store.index_at_time(float(t))
    if idx is None:
        return {"ok": False, "error": "no events at or before requested time",
                "tool": "jump_to_time", "args": {"t": t}}
    ev = store.events[idx]
    return {
        "ok":   True,
        "tool": "jump_to_time",
        "args": {"t": t},
        "result": {
            "matched_t":     ev.get("t"),
            "matched_step":  ev.get("step"),
            "delta_seconds": (float(t) - float(ev.get("t", t))),
            "event":         ev,
        },
    }


# ============================================================
# Tool 2: interval_stats
# ============================================================

def interval_stats(store: EventStore, t1: float, t2: float,
                   fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Aggregate stats for numeric fields in [t1, t2]."""
    store.reload_if_changed()
    rows = store.slice_by_time(float(t1), float(t2))
    if fields is None:
        fields = _KPI_FIELDS + ["action.A_avg_scale"]

    summary: Dict[str, Any] = {}
    for f in fields:
        vals = [float(r[f]) for r in rows if f in r and isinstance(r[f], (int, float))]
        summary[f] = _stats(vals)

    # B_mode is discrete -> counts
    bmode_counts: Dict[str, int] = {}
    for r in rows:
        if "action.B_mode" in r:
            k = str(int(r["action.B_mode"]))
            bmode_counts[k] = bmode_counts.get(k, 0) + 1

    return {
        "ok":   True,
        "tool": "interval_stats",
        "args": {"t1": t1, "t2": t2, "fields": fields},
        "result": {
            "n_events":     len(rows),
            "first_step":   rows[0].get("step") if rows else None,
            "last_step":    rows[-1].get("step") if rows else None,
            "summary":      summary,
            "B_mode_counts": bmode_counts,
        },
    }


# ============================================================
# Tool 3: find_mode_changes
# ============================================================

def find_mode_changes(store: EventStore, t1: float, t2: float,
                      field: str = "action.B_mode") -> Dict[str, Any]:
    """List every index where `field` changes value in [t1, t2]."""
    store.reload_if_changed()
    rows = store.slice_by_time(float(t1), float(t2))

    changes = []
    prev = None
    for r in rows:
        if field not in r:
            continue
        v = r[field]
        if prev is not None and v != prev[0]:
            changes.append({
                "t":        r.get("t"),
                "step":     r.get("step"),
                "from":     prev[0],
                "to":       v,
                "from_step": prev[1],
            })
        prev = (v, r.get("step"))

    return {
        "ok":   True,
        "tool": "find_mode_changes",
        "args": {"t1": t1, "t2": t2, "field": field},
        "result": {
            "n_changes": len(changes),
            "changes":   changes,
        },
    }


# ============================================================
# Tool 4: explain_change
# ============================================================

def explain_change(store: EventStore, t: float, window: int = 5) -> Dict[str, Any]:
    """
    Look at the event nearest to time t. Compare KPI/state averages over the
    `window` events before vs after. Also dump the LMUT rules active at t.
    """
    store.reload_if_changed()
    idx = store.index_at_time(float(t))
    if idx is None:
        return {"ok": False, "tool": "explain_change", "args": {"t": t, "window": window},
                "error": "no events at or before requested time"}

    evs = store.events
    lo_before = max(0, idx - window)
    hi_after  = min(len(evs), idx + 1 + window)
    before = evs[lo_before:idx]
    after  = evs[idx + 1:hi_after]

    def avg(rows, k):
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        return statistics.fmean(vals) if vals else None

    fields = _KPI_FIELDS + ["action.A_avg_scale"]
    diff: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        b = avg(before, f)
        a = avg(after,  f)
        if b is None and a is None:
            continue
        delta = None
        if b is not None and a is not None:
            delta = a - b
        diff[f] = {"before_mean": b, "after_mean": a, "delta": delta}

    pivot = evs[idx]
    b_mode_change = None
    if before and "action.B_mode" in before[-1] and "action.B_mode" in pivot:
        if before[-1]["action.B_mode"] != pivot["action.B_mode"]:
            b_mode_change = {"from": before[-1]["action.B_mode"],
                             "to":   pivot["action.B_mode"]}

    return {
        "ok":   True,
        "tool": "explain_change",
        "args": {"t": t, "window": window},
        "result": {
            "pivot_t":          pivot.get("t"),
            "pivot_step":       pivot.get("step"),
            "B_mode_change":    b_mode_change,
            "before_n":         len(before),
            "after_n":          len(after),
            "kpi_deltas":       diff,
            "lmut_rules_A":     pivot.get("lmut.rules_A_top", []),
            "lmut_rules_B":     pivot.get("lmut.rules_B_top", []),
            "lmut_acc_B":       pivot.get("lmut.acc_B"),
            "lmut_coverage":    pivot.get("lmut.coverage"),
            "lmut_precision":   pivot.get("lmut.precision"),
        },
    }


# ============================================================
# Tool 5: top_rules_at
# ============================================================

def top_rules_at(store: EventStore, t: float, k: int = 5) -> Dict[str, Any]:
    store.reload_if_changed()
    idx = store.index_at_time(float(t))
    if idx is None:
        return {"ok": False, "tool": "top_rules_at", "args": {"t": t, "k": k},
                "error": "no events at or before requested time"}
    ev = store.events[idx]
    return {
        "ok":   True,
        "tool": "top_rules_at",
        "args": {"t": t, "k": k},
        "result": {
            "t":     ev.get("t"),
            "step":  ev.get("step"),
            "rules_A_top": (ev.get("lmut.rules_A_top") or [])[:k],
            "rules_B_top": (ev.get("lmut.rules_B_top") or [])[:k],
            "stability":   ev.get("lmut.stability"),
            "coverage":    ev.get("lmut.coverage"),
            "precision":   ev.get("lmut.precision"),
        },
    }


# ============================================================
# Tool 6 (bonus): run_summary
# ============================================================

def run_summary(store: EventStore) -> Dict[str, Any]:
    store.reload_if_changed()
    n = store.count()
    if n == 0:
        return {"ok": False, "tool": "run_summary", "args": {}, "error": "no events"}
    first = store.events[0]
    last  = store.events[-1]
    manifest = store.manifest()
    return {
        "ok": True,
        "tool": "run_summary",
        "args": {},
        "result": {
            "run_id":         manifest.get("run_id"),
            "schema_version": manifest.get("schema_version"),
            "n_events":       n,
            "first_t":        first.get("t"),
            "last_t":         last.get("t"),
            "first_step":     first.get("step"),
            "last_step":      last.get("step"),
            "events_written": manifest.get("events_written"),
            "rejects":        manifest.get("rejects"),
            "last_ok_ts":     manifest.get("last_ok_ts"),
        },
    }


# ----------------------------- registry -----------------------------

# Used by the LLM tool dispatcher.
TOOLS = {
    "jump_to_time":       jump_to_time,
    "interval_stats":     interval_stats,
    "find_mode_changes":  find_mode_changes,
    "explain_change":     explain_change,
    "top_rules_at":       top_rules_at,
    "run_summary":        run_summary,
}


TOOL_SCHEMAS = [
    {
        "name": "jump_to_time",
        "description": "Return the full harmonised event snapshot at (or just before) unix time t.",
        "input_schema": {
            "type": "object",
            "properties": {"t": {"type": "number", "description": "Unix timestamp."}},
            "required": ["t"],
        },
    },
    {
        "name": "interval_stats",
        "description": "Aggregate stats (mean/std/min/max/p50/p90/p99 and B_mode counts) for events in [t1, t2].",
        "input_schema": {
            "type": "object",
            "properties": {
                "t1": {"type": "number"},
                "t2": {"type": "number"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional. Default = all KPIs + A_avg_scale."},
            },
            "required": ["t1", "t2"],
        },
    },
    {
        "name": "find_mode_changes",
        "description": "List every B_mode change in [t1, t2].",
        "input_schema": {
            "type": "object",
            "properties": {
                "t1": {"type": "number"},
                "t2": {"type": "number"},
                "field": {"type": "string", "description": "Default 'action.B_mode'."},
            },
            "required": ["t1", "t2"],
        },
    },
    {
        "name": "explain_change",
        "description": "Compare KPI averages in the `window` events before vs after time t, plus active LMUT rules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "t":      {"type": "number"},
                "window": {"type": "integer", "description": "How many events on each side. Default 5."},
            },
            "required": ["t"],
        },
    },
    {
        "name": "top_rules_at",
        "description": "Return the top-k LMUT rules active at time t.",
        "input_schema": {
            "type": "object",
            "properties": {
                "t": {"type": "number"},
                "k": {"type": "integer", "description": "Default 5."},
            },
            "required": ["t"],
        },
    },
    {
        "name": "run_summary",
        "description": "High-level summary of the current run (n_events, time range, schema version).",
        "input_schema": {"type": "object", "properties": {}},
    },
]
