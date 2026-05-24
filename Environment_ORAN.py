# Environment_multicell_wrap.py
# ------------------------------------------------------------
# Multi-cell RAN environment with wrap-around (torus) geometry
# KPI/reward are computed ONLY for the center cell.
#
# API:
#   env = MultiCellWrapEnv(...)
#   obs, info = env.reset(seed=0)
#   obs, reward, done, info = env.step(a_cont, b_mode)
#
# Notes:
# - a_cont: np.ndarray shape (n_cells,), continuous in [0,1] (internally mapped to [p_min, p_max])
# - b_mode: int in {0,1,2,3}, affects scheduling policy of CENTER cell only.
#
# ------------------------------------------------------------

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # Geometry
    Lx: float = 500.0
    Ly: float = 500.0
    n_side: int = 3                    # total cells = n_side^2; center cell exists if n_side is odd
    ue_per_cell: int = 20              # total UEs = ue_per_cell * n_cells

    # Radio
    fc_ghz: float = 3.5
    tx_power_dbm_ref: float = 43.0     # reference per-cell TX power (dBm) before a_cont scaling
    p_scale_min: float = 0.5           # maps a_cont in [0,1] -> [p_scale_min, p_scale_max]
    p_scale_max: float = 1.0
    noise_figure_db: float = 7.0
    bandwidth_hz: float = 20e6
    sinr_outage_db: float = -3.0       # outage threshold (dB)
    se_cap_bpshz: float = 7.0          # cap spectral efficiency (bps/Hz)

    # Time / traffic (window-based)
    step_ms: float = 10.0              # one environment step corresponds to step_ms milliseconds
    pkt_arrival_mbps: float = 2.0      # mean arrival rate per UE (Mb/s)
    queue_cap_bits: float = 10e6       # cap queue to avoid explosion (bits)
    deadline_ms: float = 50.0          # HOL deadline for "deadline miss" metric
    hol_cap_ms: float = 200.0          # cap reported HOL for visualization stability (does not change internal logic)

    # Episode
    max_steps: int = 2000

    # Reward weights (center cell only)
    w_thr: float = 1.0
    w_outage: float = 2.0
    w_hol: float = 0.5
    w_drop: float = 1.0
    w_power: float = 0.05              # penalty on avg TX power scaling (all cells)

    # Misc
    seed: int = 0
    fast_math_eps: float = 1e-12


class MultiCellWrapEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        assert cfg.n_side >= 1 and (cfg.n_side % 2 == 1), "n_side must be odd so a unique center cell exists."
        self.n_cells = cfg.n_side * cfg.n_side
        self.center_idx = (cfg.n_side // 2) * cfg.n_side + (cfg.n_side // 2)

        # RNG
        self.rng = np.random.default_rng(cfg.seed)

        # Precompute cell positions on a grid over [0,Lx)×[0,Ly)
        # Uniform grid spacing
        xs = (np.arange(cfg.n_side) + 0.5) * (cfg.Lx / cfg.n_side)
        ys = (np.arange(cfg.n_side) + 0.5) * (cfg.Ly / cfg.n_side)
        self.cell_pos = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)  # (n_cells,2)

        # State (will be initialized in reset)
        self.step_count = 0
        self.ue_pos = None              # (n_ues,2)
        self.ue_cell = None             # serving cell index for each UE
        self.queue_bits = None          # (n_ues,)
        self.hol_ms = None              # (n_ues,) head-of-line waiting time
        self.last_sinr_db = None         # (n_ues,)

        # Convenience
        self.n_ues = self.n_cells * cfg.ue_per_cell

    # ----------------------------
    # Geometry helpers (torus)
    # ----------------------------
    def _torus_delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        a: (...,2), b: (...,2) -> delta in torus with minimum-image convention
        """
        dx = a[..., 0] - b[..., 0]
        dy = a[..., 1] - b[..., 1]
        dx -= self.cfg.Lx * np.round(dx / self.cfg.Lx)
        dy -= self.cfg.Ly * np.round(dy / self.cfg.Ly)
        return np.stack([dx, dy], axis=-1)

    def _torus_dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = self._torus_delta(a, b)
        return np.sqrt(np.sum(d * d, axis=-1) + self.cfg.fast_math_eps)

    # ----------------------------
    # Radio model
    # ----------------------------
    def _fspl_db(self, d_m: np.ndarray) -> np.ndarray:
        # Free-space path loss (approx): 32.45 + 20log10(fc_MHz) + 20log10(d_km)
        fc_mhz = self.cfg.fc_ghz * 1000.0
        d_km = d_m / 1000.0
        return 32.45 + 20.0 * np.log10(fc_mhz + self.cfg.fast_math_eps) + 20.0 * np.log10(d_km + self.cfg.fast_math_eps)

    def _noise_power_dbm(self) -> float:
        # Thermal noise: -174 dBm/Hz + 10log10(B) + NF
        return -174.0 + 10.0 * math.log10(self.cfg.bandwidth_hz) + self.cfg.noise_figure_db

    def _power_scale(self, a_cont: np.ndarray) -> np.ndarray:
        a = np.clip(a_cont, 0.0, 1.0)
        return self.cfg.p_scale_min + a * (self.cfg.p_scale_max - self.cfg.p_scale_min)

    # ----------------------------
    # Environment API
    # ----------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # UE positions: uniform over full torus region
        self.ue_pos = np.stack([
            self.rng.uniform(0.0, self.cfg.Lx, size=self.n_ues),
            self.rng.uniform(0.0, self.cfg.Ly, size=self.n_ues)
        ], axis=1).astype(np.float64)

        # Initial queues and HOL
        self.queue_bits = np.zeros(self.n_ues, dtype=np.float64)
        self.hol_ms = np.zeros(self.n_ues, dtype=np.float64)
        self.last_sinr_db = np.zeros(self.n_ues, dtype=np.float64)

        # Initial association (best received power ignoring interference)
        self.ue_cell = self._associate_ues(initial=True)

        obs = self._make_obs(center_only=True)
        info = {"center_cell": self.center_idx}
        return obs, info

    def step(self, a_cont: np.ndarray, b_mode: int):
        cfg = self.cfg
        self.step_count += 1

        # Map action to per-cell power scaling
        if a_cont is None:
            a_cont = np.zeros(self.n_cells, dtype=np.float64)
        a_cont = np.asarray(a_cont, dtype=np.float64).reshape(-1)
        if a_cont.shape[0] != self.n_cells:
            # allow broadcast if user passes a scalar
            if a_cont.shape[0] == 1:
                a_cont = np.repeat(a_cont.item(), self.n_cells)
            else:
                raise ValueError(f"a_cont must have shape ({self.n_cells},) or (1,), got {a_cont.shape}")
        p_scale = self._power_scale(a_cont)

        # Update association each step (optional); this tends to stabilize interference realism
        self.ue_cell = self._associate_ues(initial=False)

        # Traffic arrivals
        arrivals_bits = self._sample_arrivals_bits()

        # Add arrivals, cap queue
        self.queue_bits = np.minimum(self.queue_bits + arrivals_bits, cfg.queue_cap_bits)

        # Compute SINR for all UEs under multi-cell interference (wrap-around distances)
        sinr_lin, sinr_db = self._compute_sinr(p_scale)
        self.last_sinr_db = sinr_db

        # Scheduling for CENTER cell only
        center_ue_idx = np.where(self.ue_cell == self.center_idx)[0]
        served_bits = np.zeros(self.n_ues, dtype=np.float64)

        if center_ue_idx.size > 0:
            served_bits_center = self._schedule_and_serve_center(center_ue_idx, sinr_lin, b_mode, p_scale[self.center_idx])
            served_bits[center_ue_idx] = served_bits_center

        # Apply service
        self.queue_bits = np.maximum(self.queue_bits - served_bits, 0.0)

        # HOL update and deadline misses / drops
        deadline_miss_bits, drops_bits = self._update_hol_and_drops(center_ue_idx)

        # KPIs (center cell only)
        kpi = self._center_kpis(center_ue_idx, sinr_db, served_bits, drops_bits, deadline_miss_bits)

        # Reward: throughput - penalties (center cell metrics) - power penalty (all cells)
        # Throughput in Mb/s over step window
        thr_mbps = (kpi["served_bits_win"] / (cfg.step_ms / 1000.0)) / 1e6
        avg_power_scale = float(np.mean(p_scale))

        reward = (
            cfg.w_thr * thr_mbps
            - cfg.w_outage * kpi["outage_ratio"]
            - cfg.w_hol * (kpi["avg_hol_ms"] / cfg.deadline_ms)
            - cfg.w_drop * (kpi["drops_bits_win"] / (cfg.queue_cap_bits + cfg.fast_math_eps))
            - cfg.w_power * avg_power_scale
        )

        done = self.step_count >= cfg.max_steps

        obs = self._make_obs(center_only=True, kpi=kpi, avg_power_scale=avg_power_scale)
        info = dict(kpi)
        info.update({
            "step": self.step_count,
            "avg_power_scale": avg_power_scale,
            "n_center_ues": int(center_ue_idx.size),
        })
        return obs, float(reward), bool(done), info

    # ----------------------------
    # Core computations
    # ----------------------------
    def _associate_ues(self, initial: bool = False) -> np.ndarray:
        """
        Associate each UE to the cell with maximum received power (pathloss only).
        """
        # distances (n_ues, n_cells)
        d = self._torus_dist(self.ue_pos[:, None, :], self.cell_pos[None, :, :])
        pl_db = self._fspl_db(d)  # (n_ues, n_cells)
        # Received power ranking: higher is better => minimize pathloss
        # (TX power is same across cells for association; interference ignored)
        best = np.argmin(pl_db, axis=1)
        return best.astype(np.int32)

    def _sample_arrivals_bits(self) -> np.ndarray:
        """
        Poisson arrivals per UE.
        mean bits = rate(Mb/s)*1e6*(step_s)
        """
        step_s = self.cfg.step_ms / 1000.0
        mean_bits = self.cfg.pkt_arrival_mbps * 1e6 * step_s
        # Poisson on bits is too granular; sample packet counts then scale.
        # Use a compound Poisson: N~Poisson(lambda), packet_size_bits~Exp(mean_pkt_bits)
        # Here: approximate by Poisson on "chunks" then scale.
        mean_pkt_bits = 12_000.0  # ~1500B
        lam = mean_bits / mean_pkt_bits
        n_pkts = self.rng.poisson(lam, size=self.n_ues)
        pkt_bits = self.rng.exponential(mean_pkt_bits, size=self.n_ues)
        return (n_pkts * pkt_bits).astype(np.float64)

    def _compute_sinr(self, p_scale: np.ndarray):
        """
        Multi-cell downlink SINR per UE under wrap-around.
        Simple model: each cell transmits full-band with power scaling.
        """
        cfg = self.cfg
        noise_dbm = self._noise_power_dbm()
        noise_mw = 10 ** (noise_dbm / 10.0)

        # Distances UE->cell: (n_ues, n_cells)
        d = self._torus_dist(self.ue_pos[:, None, :], self.cell_pos[None, :, :])
        pl_db = self._fspl_db(d)

        # TX power per cell in dBm (scaled in linear domain)
        # scale acts as linear scaling; convert via +10log10(scale)
        tx_dbm = cfg.tx_power_dbm_ref + 10.0 * np.log10(p_scale + cfg.fast_math_eps)

        # Received power in mW: Pr_mw = 10^((tx_dbm - pl_db)/10)
        pr_mw = 10 ** ((tx_dbm[None, :] - pl_db) / 10.0)  # (n_ues, n_cells)

        serving = self.ue_cell
        signal = pr_mw[np.arange(self.n_ues), serving]          # (n_ues,)
        interf = np.sum(pr_mw, axis=1) - signal                 # (n_ues,)

        sinr_lin = signal / (interf + noise_mw + cfg.fast_math_eps)
        sinr_db = 10.0 * np.log10(sinr_lin + cfg.fast_math_eps)
        return sinr_lin, sinr_db

    def _schedule_and_serve_center(self, center_ue_idx: np.ndarray, sinr_lin: np.ndarray, b_mode: int, p_scale_center: float):
        """
        Allocate the center cell bandwidth across its associated UEs and compute served bits in this step.
        b_mode:
          0: Round-robin (equal share)
          1: Proportional fair (share ∝ log(1+sinr))
          2: HOL-priority (share ∝ HOL)
          3: Max-SINR (share ∝ sinr)
        """
        cfg = self.cfg
        step_s = cfg.step_ms / 1000.0
        B = cfg.bandwidth_hz

        idx = center_ue_idx
        sinr = sinr_lin[idx]

        # Spectral efficiency (bps/Hz) with cap
        se = np.log2(1.0 + sinr)
        se = np.minimum(se, cfg.se_cap_bpshz)

        # weights per mode
        if b_mode == 0:
            w = np.ones_like(se)
        elif b_mode == 1:
            w = np.log(1.0 + sinr + cfg.fast_math_eps)
        elif b_mode == 2:
            w = np.maximum(self.hol_ms[idx], 0.0) + 1.0
        elif b_mode == 3:
            w = sinr + cfg.fast_math_eps
        else:
            w = np.ones_like(se)

        w_sum = float(np.sum(w)) + cfg.fast_math_eps
        share = w / w_sum

        # Served bits for UE i: share_i * B * se_i * step_s
        served = share * B * se * step_s

        # cannot serve more than queue
        served = np.minimum(served, self.queue_bits[idx])
        return served.astype(np.float64)

    def _update_hol_and_drops(self, center_ue_idx: np.ndarray):
        """
        Update HOL time for all UEs; compute deadline misses and drops for center UEs only.
        """
        cfg = self.cfg
        # HOL increases if queue>0, else resets
        active = self.queue_bits > 0.0
        self.hol_ms[active] += cfg.step_ms
        self.hol_ms[~active] = 0.0

        deadline_miss_bits = 0.0
        drops_bits = 0.0

        if center_ue_idx.size > 0:
            idx = center_ue_idx
            # Deadline miss: HOL > deadline
            miss = self.hol_ms[idx] > cfg.deadline_ms
            if np.any(miss):
                # Mark bits as missed proportionally to queue (simple proxy)
                miss_bits = float(np.sum(self.queue_bits[idx][miss]) * 0.05)  # 5% of queued bits counted as "miss" per step
                deadline_miss_bits += miss_bits

            # Drops: if HOL exceeds 2*deadline, drop some bits
            drop = self.hol_ms[idx] > (2.0 * cfg.deadline_ms)
            if np.any(drop):
                drop_bits = self.queue_bits[idx][drop] * 0.10  # drop 10% of queue
                self.queue_bits[idx][drop] = np.maximum(self.queue_bits[idx][drop] - drop_bits, 0.0)
                drops_bits += float(np.sum(drop_bits))

        return float(deadline_miss_bits), float(drops_bits)

    def _center_kpis(self, center_ue_idx, sinr_db, served_bits, drops_bits, deadline_miss_bits):
        cfg = self.cfg
        if center_ue_idx.size == 0:
            return {
                "avg_se": 0.0,
                "avg_sinr_db": 0.0,
                "outage_ratio": 0.0,
                "queue_ratio": 0.0,
                "avg_hol_ms": 0.0,
                "drops_bits_win": float(drops_bits),
                "served_bits_win": 0.0,
                "deadline_miss_win": float(deadline_miss_bits),
            }

        idx = center_ue_idx
        # SE from last SINR
        sinr_lin = 10 ** (sinr_db[idx] / 10.0)
        se = np.log2(1.0 + sinr_lin)
        se = np.minimum(se, cfg.se_cap_bpshz)

        outage_ratio = float(np.mean(sinr_db[idx] < cfg.sinr_outage_db))
        queue_ratio = float(np.mean(self.queue_bits[idx] / (cfg.queue_cap_bits + cfg.fast_math_eps)))
        avg_hol_ms = float(np.mean(np.minimum(self.hol_ms[idx], cfg.hol_cap_ms)))
        served_bits_win = float(np.sum(served_bits[idx]))

        return {
            "avg_se": float(np.mean(se)),
            "avg_sinr_db": float(np.mean(sinr_db[idx])),
            "outage_ratio": outage_ratio,
            "queue_ratio": queue_ratio,
            "avg_hol_ms": avg_hol_ms,
            "drops_bits_win": float(drops_bits),
            "served_bits_win": served_bits_win,
            "deadline_miss_win": float(deadline_miss_bits),
        }

    def _make_obs(self, center_only: bool = True, kpi: dict | None = None, avg_power_scale: float | None = None):
        """
        Fixed-length observation vector suitable for an MLP.
        """
        cfg = self.cfg
        if kpi is None:
            # compute from current state with nominal power scale = 1 for obs (or reuse last sinr)
            center_ue_idx = np.where(self.ue_cell == self.center_idx)[0]
            kpi = self._center_kpis(center_ue_idx, self.last_sinr_db, np.zeros(self.n_ues), 0.0, 0.0)

        if avg_power_scale is None:
            avg_power_scale = 0.0

        # Normalized features (keep stable ranges)
        # You can append more fields if your agent expects a wider vector.
        obs = np.array([
            kpi["avg_sinr_db"] / 30.0,                    # ~[-, +] normalized
            kpi["avg_se"] / cfg.se_cap_bpshz,             # [0,1]
            kpi["outage_ratio"],                          # [0,1]
            kpi["queue_ratio"],                           # [0,1]
            min(kpi["avg_hol_ms"], cfg.hol_cap_ms) / cfg.hol_cap_ms,  # [0,1]
            (kpi["served_bits_win"] / (cfg.bandwidth_hz * (cfg.step_ms/1000.0) * cfg.se_cap_bpshz + cfg.fast_math_eps)),  # [0,~1]
            (kpi["drops_bits_win"] / (cfg.queue_cap_bits + cfg.fast_math_eps)),  # [0,1]
            (kpi["deadline_miss_win"] / (cfg.queue_cap_bits + cfg.fast_math_eps)),  # [0,1]
            avg_power_scale,                               # ~[0.5,1.0] if you pass it
            float(self.step_count) / float(cfg.max_steps), # [0,1]
        ], dtype=np.float64)

        return obs

    # Optional: expose a simple render hook (no-op)
    def render(self):
        pass


# ----------------------------
# Quick self-test
# ----------------------------
if __name__ == "__main__":
    cfg = EnvConfig(Lx=500, Ly=500, n_side=3, ue_per_cell=10, max_steps=50)
    env = MultiCellWrapEnv(cfg)
    obs, info = env.reset(seed=1)
    print("obs shape:", obs.shape, "center:", info["center_cell"])

    for t in range(5):
        a = np.ones(env.n_cells) * 0.5
        b = t % 4
        obs, r, done, info = env.step(a, b)
        print(t, "r=", round(r, 3), "SINR=", round(info["avg_sinr_db"], 2), "outage=", round(info["outage_ratio"], 3),
              "HOL=", round(info["avg_hol_ms"], 2), "served=", int(info["served_bits_win"]))
