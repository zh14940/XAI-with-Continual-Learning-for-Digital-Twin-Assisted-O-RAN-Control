"""
logger_harmoniser.py  --  D1: Telemetry & Logging Hardening
-----------------------------------------------------------
Reads:
  - marl_rollout_log.csv   (per-step rollout from xApp1/xApp2)
  - demo_status.json       (current snapshot incl. LMUT diag/rules)

Writes:
  - runs/<run_id>/events.jsonl   (one harmonised event per line)
  - runs/<run_id>/manifest.json  (schema_version, counts, start/end, last_ok_ts)
  - runs/<run_id>/heartbeat.json (updated every loop; lets dashboards detect stalls)

Design goals:
  * Stable schema (validated against schema.json on every write).
  * Tail-follow the CSV; survive restarts via a byte offset stored in manifest.
  * Atomic writes; never half-write a line.
  * If status.json has fresher LMUT fields than the CSV row, merge them in.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List


# ----------------------------- IO helpers -----------------------------

def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: str, payload: dict, retry: int = 10) -> None:
    tmp = path + ".tmp"
    for _ in range(retry):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    # last-resort direct write
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: str, records: List[dict]) -> None:
    if not records:
        return
    # Open in append-binary so partial writes can't corrupt earlier lines.
    with open(path, "ab") as f:
        for r in records:
            line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


# ----------------------------- Schema --------------------------------

def load_schema(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_TYPE_CHECKS = {
    "float":       lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "int":         lambda v: isinstance(v, int) and not isinstance(v, bool),
    "string":      lambda v: isinstance(v, str),
    "list[string]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
}


def validate_event(event: dict, schema: dict) -> Tuple[bool, str]:
    fields = schema["fields"]
    for fname, fdef in fields.items():
        if fdef.get("required") and fname not in event:
            return False, f"missing required field: {fname}"
        if fname in event and event[fname] is not None:
            t = fdef["type"]
            check = _TYPE_CHECKS.get(t)
            if check and not check(event[fname]):
                return False, f"field {fname} has wrong type, expected {t}"
    return True, ""


# ----------------------------- CSV tail ------------------------------

def tail_csv(path: str, last_pos: int) -> Tuple[List[dict], int, Optional[List[str]]]:
    """Read new rows since last_pos. Returns (rows, new_pos, header)."""
    if not os.path.exists(path):
        return [], last_pos, None

    rows: List[dict] = []
    header: Optional[List[str]] = None

    with open(path, "rb") as f:
        # If file shrunk (rotated), reset.
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if last_pos > end:
            last_pos = 0

        # Always read header from start (cheap).
        f.seek(0)
        first_line = f.readline().decode("utf-8", errors="replace").rstrip("\r\n")
        if first_line:
            header = first_line.split(",")

        if last_pos == 0:
            last_pos = f.tell()  # skip header

        f.seek(last_pos)
        for line in f:
            try:
                txt = line.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                continue
            if not txt:
                continue
            parts = txt.split(",")
            if header and len(parts) == len(header):
                rows.append(dict(zip(header, parts)))
        new_pos = f.tell()

    return rows, new_pos, header


# ----------------------------- Harmonisation -------------------------

def harmonise_row(row: dict, status: dict, run_id: str, step_fallback: int) -> dict:
    """Convert a raw CSV row + current status snapshot into a harmonised event."""
    # Step
    step = _safe_int(row.get("iter") or row.get("step"), default=step_fallback)

    # KPI (CSV is source of truth per-step; status is a snapshot fallback)
    def pick(*keys, kind="float"):
        for k in keys:
            if k in row and row[k] not in ("", None):
                return _safe_float(row[k]) if kind == "float" else _safe_int(row[k])
        return None

    # Per real csv header: ts_unix,iter,avg_se,avg_sinr_db,outage_ratio,
    #                      queue_ratio,avg_hol_ms,drops_bits_win,A_avg_scale,B_mode
    kpi_se      = pick("avg_se", "se", "spectral_efficiency")
    kpi_hol     = pick("avg_hol_ms", "hol_ms", "kpi.avg_hol_ms")
    kpi_outage  = pick("outage_ratio", "kpi.outage_ratio")
    kpi_sinr    = pick("avg_sinr_db", "sinr_db", "kpi.avg_sinr_db")
    kpi_queue   = pick("queue_ratio", "kpi.queue_ratio")
    kpi_drops   = pick("drops_bits_win", "kpi.drops_bits_win")

    act_a = pick("A_avg_scale", "action_A", "a_scale")
    act_b = pick("B_mode", "action_B", "b_mode", kind="int")

    # Use ts_unix from csv when present, else wall-clock now.
    ts_csv = _safe_float(row.get("ts_unix"))

    # LMUT fields: pull from status snapshot (csv has no LMUT cols).
    lmut_meta = status.get("meta", {}) if isinstance(status.get("meta"), dict) else {}
    rel = status.get("reliability", {}) if isinstance(status.get("reliability"), dict) else {}
    diag = status.get("diag", {}) if isinstance(status.get("diag"), dict) else {}
    rules = status.get("rules", {}) if isinstance(status.get("rules"), dict) else {}

    event = {
        "t":      ts_csv if ts_csv is not None else time.time(),
        "run_id": run_id,
        "source": "harmoniser",
        "step":   int(step) if step is not None else step_fallback,
    }

    if kpi_se     is not None: event["kpi.avg_se"]         = float(kpi_se)
    if kpi_hol    is not None: event["kpi.avg_hol_ms"]     = float(kpi_hol)
    if kpi_outage is not None: event["kpi.outage_ratio"]   = float(kpi_outage)
    if kpi_sinr   is not None: event["kpi.avg_sinr_db"]    = float(kpi_sinr)
    if kpi_queue  is not None: event["kpi.queue_ratio"]    = float(kpi_queue)
    if kpi_drops  is not None: event["kpi.drops_bits_win"] = float(kpi_drops)

    if act_a is not None: event["action.A_avg_scale"] = float(act_a)
    if act_b is not None: event["action.B_mode"]      = int(act_b)

    if rel:
        if "acc_B"      in rel: event["lmut.acc_B"]      = _safe_float(rel["acc_B"])
        if "baseline_B" in rel: event["lmut.baseline_B"] = _safe_float(rel["baseline_B"])
        if "mae_A"      in rel: event["lmut.mae_A"]      = _safe_float(rel["mae_A"])
    if diag:
        if "coverage"  in diag: event["lmut.coverage"]  = _safe_float(diag["coverage"])
        if "precision" in diag: event["lmut.precision"] = _safe_float(diag["precision"])
        if "stability" in diag: event["lmut.stability"] = _safe_float(diag["stability"])
    if "tau_cov" in lmut_meta:
        event["lmut.tau_cov"] = _safe_float(lmut_meta["tau_cov"])
    if isinstance(rules.get("A_rules_top"), list):
        event["lmut.rules_A_top"] = [str(x) for x in rules["A_rules_top"]]
    if isinstance(rules.get("B_rules_top"), list):
        event["lmut.rules_B_top"] = [str(x) for x in rules["B_rules_top"]]

    # Strip Nones
    return {k: v for k, v in event.items() if v is not None}


# ----------------------------- Main loop -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default="marl_rollout_log.csv")
    ap.add_argument("--status",    default="demo_status.json")
    ap.add_argument("--schema",    default="schema.json")
    ap.add_argument("--out_dir",   default="runs")
    ap.add_argument("--run_id",    default=None, help="Override; default = timestamp.")
    ap.add_argument("--sleep_s",   type=float, default=1.0)
    ap.add_argument("--heartbeat_every", type=float, default=2.0)
    args = ap.parse_args()

    schema = load_schema(args.schema)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    events_path    = str(run_dir / "events.jsonl")
    manifest_path  = str(run_dir / "manifest.json")
    heartbeat_path = str(run_dir / "heartbeat.json")

    # Restore previous state if manifest exists (restart resilience).
    manifest = _read_json_safe(manifest_path)
    if not manifest:
        manifest = {
            "run_id":          run_id,
            "schema_version":  schema["schema_version"],
            "schema_path":     os.path.abspath(args.schema),
            "csv_source":      os.path.abspath(args.csv),
            "status_source":   os.path.abspath(args.status),
            "started_at":      time.time(),
            "events_written":  0,
            "csv_byte_offset": 0,
            "last_ok_ts":      None,
            "last_step":       None,
            "rejects":         0,
        }
        _atomic_write_json(manifest_path, manifest)

    last_pos = int(manifest.get("csv_byte_offset", 0))
    step_counter = int(manifest.get("last_step") or 0)
    last_heartbeat = 0.0

    print(f"[harmoniser] run_id={run_id}", flush=True)
    print(f"[harmoniser] tailing {args.csv} from byte {last_pos}", flush=True)
    print(f"[harmoniser] writing -> {events_path}", flush=True)

    while True:
        try:
            rows, new_pos, _ = tail_csv(args.csv, last_pos)
            status = _read_json_safe(args.status)

            accepted: List[dict] = []
            rejected = 0
            for row in rows:
                step_counter += 1
                ev = harmonise_row(row, status, run_id, step_fallback=step_counter)
                ok, err = validate_event(ev, schema)
                if ok:
                    accepted.append(ev)
                else:
                    rejected += 1
                    # Don't crash; record to manifest.
                    print(f"[harmoniser][reject] step={step_counter}: {err}", flush=True)

            if accepted:
                _append_jsonl(events_path, accepted)

            last_pos = new_pos
            manifest["csv_byte_offset"] = last_pos
            manifest["events_written"] += len(accepted)
            manifest["rejects"]        += rejected
            manifest["last_ok_ts"]      = time.time()
            manifest["last_step"]       = step_counter
            _atomic_write_json(manifest_path, manifest)

            now = time.time()
            if now - last_heartbeat >= args.heartbeat_every:
                _atomic_write_json(heartbeat_path, {
                    "ts": now,
                    "alive": True,
                    "events_written": manifest["events_written"],
                    "last_step": manifest["last_step"],
                })
                last_heartbeat = now

            time.sleep(args.sleep_s)

        except KeyboardInterrupt:
            print("[harmoniser] stopping (KeyboardInterrupt)", flush=True)
            break
        except Exception as e:
            # Never die. Long runs need this.
            print(f"[harmoniser][error] {type(e).__name__}: {e}", flush=True)
            time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()