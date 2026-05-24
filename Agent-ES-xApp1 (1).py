import os
import csv
import time
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Environment_ORAN import MultiCellWrapEnv, EnvConfig


@dataclass
class Config:
    seed: int = 7
    device: str = "cpu"

    # total_steps = -1 means run forever
    total_steps: int = -1

    sync_interval: int = 10
    keep_history: int = 200
    print_interval: int = 200

    log_csv: str = r"G:\yoran_rl\marl_rollout_log.csv"
    status_json: str = r"G:\yoran_rl\demo_status.json"

    # ---- scenario ----
    Lx: float = 1500.0
    Ly: float = 1500.0
    n_side: int = 3
    ue_per_cell: int = 15

    env_max_steps: int = 1_000_000
    reset_every_steps: int = 5000
    fixed_reset_seed: bool = True
    reset_seed: int = 123

    # ---- env knobs ----
    p_scale_min: float = 0.2
    p_scale_max: float = 1.5
    sinr_outage_db: float = 0
    bandwidth_hz: float = 20e6
    pkt_arrival_mbps: float = 2.2
    deadline_ms: float = 150.0
    hol_cap_ms: float = 500.0

    # ---- learnable ICIC (deterministic) ----
    center_boost: float = 0.35
    neighbor_factor: float = 0.08

    # ---- PPO ----
    rollout_len: int = 512
    ppo_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    a_log_std_init: float = -0.6

    # ---- shaped reward weights ----
    w_thr: float = 1.0
    w_sinr: float = 1.0
    w_out: float = 10.0
    w_hol: float = 0.8
    w_pow: float = 0.3

    # ---- IO flush cadence ----
    csv_flush_every: int = 50  # steps
    json_retry: int = 15       # attempts


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _to_vec(obs):
    if isinstance(obs, dict):
        return np.array([float(v) for v in obs.values()], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _atomic_write_json(path: str, payload: dict, retry: int = 15):
    tmp = path + ".tmp"
    for _ in range(retry):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
        except Exception:
            time.sleep(0.05)
    # last resort
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _merge_dashboard(existing: dict, patch: dict) -> dict:
    if not isinstance(existing, dict):
        existing = {}
    out = dict(existing)

    # merge known top-level keys
    for k in ["meta", "kpi", "action", "history", "dist", "reliability", "rules", "diag"]:
        if k in patch:
            out[k] = patch[k]
        else:
            out[k] = out.get(k, {} if k in ["reliability", "rules", "diag"] else out.get(k))

    return out


class ActorA(nn.Module):
    """2-dim squashed Gaussian: [a_center_01, a_nei_01]."""
    def __init__(self, obs_dim: int, log_std_init: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.log_std = nn.Parameter(torch.ones(2) * log_std_init)

    def forward(self, obs: torch.Tensor):
        mu = self.net(obs)
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std


class ActorB(nn.Module):
    def __init__(self, obs_dim: int, n_modes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_modes),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs).squeeze(-1)


def _squashed_gaussian_sample(mu, std):
    eps = torch.randn_like(mu)
    z = mu + std * eps
    a = torch.sigmoid(z)

    var = std * std
    logp_z = -0.5 * (((z - mu) ** 2) / (var + 1e-8) + 2.0 * torch.log(std + 1e-8) + np.log(2.0 * np.pi))
    logp_z = logp_z.sum(dim=-1)

    log_det = torch.log(a * (1.0 - a) + 1e-8).sum(dim=-1)
    logp_a = logp_z - log_det
    return a, z, logp_a


def _squashed_gaussian_logp(mu, std, a):
    z = torch.log(a / (1.0 - a + 1e-8) + 1e-8)

    var = std * std
    logp_z = -0.5 * (((z - mu) ** 2) / (var + 1e-8) + 2.0 * torch.log(std + 1e-8) + np.log(2.0 * np.pi))
    logp_z = logp_z.sum(dim=-1)

    log_det = torch.log(a * (1.0 - a) + 1e-8).sum(dim=-1)
    return logp_z - log_det


class RolloutBuffer:
    def __init__(self, obs_dim, T, device):
        self.device = device
        self.T = T
        self.ptr = 0

        self.obs = torch.zeros((T, obs_dim), device=device)
        self.a2 = torch.zeros((T, 2), device=device)
        self.b = torch.zeros((T,), dtype=torch.long, device=device)

        self.logp_a = torch.zeros((T,), device=device)
        self.logp_b = torch.zeros((T,), device=device)
        self.v = torch.zeros((T,), device=device)
        self.r = torch.zeros((T,), device=device)
        self.done = torch.zeros((T,), device=device)

        self.adv = torch.zeros((T,), device=device)
        self.ret = torch.zeros((T,), device=device)

    def add(self, obs, a2, b, logp_a, logp_b, v, r, done):
        i = self.ptr
        self.obs[i] = obs.detach()
        self.a2[i] = a2.detach()
        self.b[i] = b.detach()
        self.logp_a[i] = logp_a.detach()
        self.logp_b[i] = logp_b.detach()
        self.v[i] = v.detach()
        self.r[i] = r.detach()
        self.done[i] = done.detach()
        self.ptr += 1

    def full(self):
        return self.ptr >= self.T

    def compute_gae(self, last_v, gamma, lam):
        adv = 0.0
        for t in reversed(range(self.T)):
            next_v = last_v if t == self.T - 1 else self.v[t + 1]
            not_done = 1.0 - self.done[t]
            delta = self.r[t] + gamma * next_v * not_done - self.v[t]
            adv = delta + gamma * lam * not_done * adv
            self.adv[t] = adv

        self.ret = self.adv + self.v
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)

    def get_minibatches(self, batch_size):
        idx = torch.randperm(self.T, device=self.device)
        for start in range(0, self.T, batch_size):
            yield idx[start:start + batch_size]

    def reset(self):
        self.ptr = 0


def build_per_cell_action(a_center_01: float, a_nei_01: float, n_cells: int, center_idx: int, cfg: Config) -> np.ndarray:
    a_center = np.clip(a_center_01 + cfg.center_boost, 0.0, 1.0)
    a_nei = np.clip(a_nei_01 * cfg.neighbor_factor, 0.0, 1.0)
    a = np.full((n_cells,), a_nei, dtype=np.float64)
    a[center_idx] = a_center
    return a


def a01_to_avg_scale(a_01: np.ndarray, env_cfg: EnvConfig) -> float:
    a_mean = float(np.mean(a_01))
    return float(env_cfg.p_scale_min + a_mean * (env_cfg.p_scale_max - env_cfg.p_scale_min))


def compute_shaped_reward(info: dict, a_avg_scale: float, cfg: Config, env_cfg: EnvConfig) -> float:
    served_bits_win = float(info.get("served_bits_win", 0.0))
    step_s = float(env_cfg.step_ms) / 1000.0
    thr_mbps = (served_bits_win / max(step_s, 1e-9)) / 1e6

    sinr_db = float(info.get("avg_sinr_db", 0.0))
    outage = float(info.get("outage_ratio", 0.0))
    hol = float(info.get("avg_hol_ms", 0.0))

    sinr_norm = np.clip(sinr_db, -10.0, 20.0) / 10.0
    hol_norm = np.clip(hol / max(cfg.deadline_ms, 1e-6), 0.0, 5.0)

    pow_norm = (a_avg_scale - cfg.p_scale_min) / max(cfg.p_scale_max - cfg.p_scale_min, 1e-6)
    pow_norm = float(np.clip(pow_norm, 0.0, 1.0))

    r = (
        cfg.w_thr * thr_mbps
        + cfg.w_sinr * sinr_norm
        - cfg.w_out * outage
        - cfg.w_hol * hol_norm
        - cfg.w_pow * pow_norm
    )
    return float(r)


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    _ensure_parent_dir(cfg.log_csv)
    _ensure_parent_dir(cfg.status_json)

    # CSV header
    header = [
        "ts_unix", "iter",
        "avg_se", "avg_sinr_db", "outage_ratio",
        "queue_ratio", "avg_hol_ms", "drops_bits_win",
        "A_avg_scale", "B_mode"
    ]
    with open(cfg.log_csv, "w", newline="", encoding="utf-8") as f_init:
        csv.writer(f_init).writerow(header)
    print(f">>> Log cleaned: {cfg.log_csv}")

    # Initialize dashboard JSON with meta so Step is never N/A
    empty_status = {
        "meta": {"timestamp": time.time(), "step": 0},
        "kpi": {"thr_bps": 0.0, "avg_hol_ms": 0.0, "outage_ratio": 0.0, "avg_sinr_db": 0.0, "queue_ratio": 0.0, "drops_bits_win": 0.0},
        "action": {"A_avg_scale": 0.0, "B_mode": 0},
        "history": {"A_avg_scale": [], "acc_B": []},
        "dist": {"B_mode_counts": {"0": 0, "1": 0, "2": 0, "3": 0}},
        "reliability": {}, "rules": {}, "diag": {},
    }
    _atomic_write_json(cfg.status_json, empty_status, retry=cfg.json_retry)
    print(">>> Dashboard JSON Reset (with meta.step).")

    env_cfg = EnvConfig(
        Lx=cfg.Lx, Ly=cfg.Ly,
        n_side=cfg.n_side,
        ue_per_cell=cfg.ue_per_cell,
        max_steps=cfg.env_max_steps,
        seed=cfg.seed,
        p_scale_min=cfg.p_scale_min,
        p_scale_max=cfg.p_scale_max,
        sinr_outage_db=cfg.sinr_outage_db,
        bandwidth_hz=cfg.bandwidth_hz,
        pkt_arrival_mbps=cfg.pkt_arrival_mbps,
        deadline_ms=cfg.deadline_ms,
        hol_cap_ms=cfg.hol_cap_ms,
    )
    env = MultiCellWrapEnv(env_cfg)

    if cfg.fixed_reset_seed:
        obs0, _ = env.reset(seed=cfg.reset_seed)
    else:
        obs0, _ = env.reset(seed=cfg.seed)
    obs = _to_vec(obs0)

    obs_dim = int(obs.shape[0])
    n_cells = int(env.n_cells)
    center_idx = int(env.center_idx)

    pi_a = ActorA(obs_dim, cfg.a_log_std_init).to(device)
    pi_b = ActorB(obs_dim, 4).to(device)
    vf = Critic(obs_dim).to(device)

    optimizer = optim.Adam(list(pi_a.parameters()) + list(pi_b.parameters()) + list(vf.parameters()), lr=cfg.lr)
    buf = RolloutBuffer(obs_dim, cfg.rollout_len, device=device)

    history_A_scale = []
    history_acc_B = []
    b_mode_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    print(f">>> System Ready | PPO=ON | Env=MultiCellWrapEnv | Status: {cfg.status_json}")

    global_step = 0

    def should_continue(step: int) -> bool:
        return (cfg.total_steps < 0) or (step < cfg.total_steps)

    with open(cfg.log_csv, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)

        while should_continue(global_step):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                v = vf(obs_t).squeeze(0)

                mu, std = pi_a(obs_t)
                a2, _z, logp_a = _squashed_gaussian_sample(mu, std)
                a2 = a2.squeeze(0)
                logp_a = logp_a.squeeze(0)

                logits = pi_b(obs_t).squeeze(0)
                dist_b = torch.distributions.Categorical(logits=logits)
                b = dist_b.sample()
                logp_b = dist_b.log_prob(b)

            a_center_01 = float(a2[0].item())
            a_nei_01 = float(a2[1].item())

            a_np = build_per_cell_action(a_center_01, a_nei_01, n_cells, center_idx, cfg)
            A_avg_scale = a01_to_avg_scale(a_np, env_cfg)

            next_obs, _env_r, done, info = env.step(a_np, int(b.item()))
            next_obs = _to_vec(next_obs)

            shaped_r = compute_shaped_reward(info, A_avg_scale, cfg, env_cfg)

            buf.add(
                obs=torch.tensor(obs, dtype=torch.float32, device=device),
                a2=a2,
                b=b,
                logp_a=logp_a,
                logp_b=logp_b,
                v=v,
                r=torch.tensor(float(shaped_r), device=device),
                done=torch.tensor(float(done), device=device),
            )

            # KPIs for logging/dashboard
            avg_se = float(info.get("avg_se", 0.0))
            avg_sinr_db = float(info.get("avg_sinr_db", 0.0))
            outage_ratio = float(info.get("outage_ratio", 0.0))
            queue_ratio = float(info.get("queue_ratio", 0.0))
            avg_hol_ms = float(info.get("avg_hol_ms", 0.0))
            drops_bits_win = float(info.get("drops_bits_win", 0.0))
            served_bits_win = float(info.get("served_bits_win", 0.0))
            n_center_ues = int(info.get("n_center_ues", -1))

            step_s = float(env_cfg.step_ms) / 1000.0
            thr_bps = served_bits_win / max(step_s, 1e-9)

            B_mode = int(b.item())
            b_mode_stats[B_mode] += 1

            # CSV row
            writer.writerow([
                time.time(), global_step,
                avg_se, avg_sinr_db, outage_ratio,
                queue_ratio, avg_hol_ms, drops_bits_win,
                A_avg_scale, float(B_mode)
            ])

            # Flush CSV periodically (safer for long runs)
            if (global_step % cfg.csv_flush_every) == 0:
                try:
                    f_csv.flush()
                except Exception:
                    pass

            # rolling history (bounded)
            history_A_scale.append(A_avg_scale)
            if len(history_A_scale) > cfg.keep_history:
                history_A_scale.pop(0)
            history_acc_B.append(0.0)
            if len(history_acc_B) > cfg.keep_history:
                history_acc_B.pop(0)

            # Dashboard sync (with meta.step)
            if global_step % cfg.sync_interval == 0:
                patch = {
                    "meta": {"timestamp": time.time(), "step": global_step},
                    "kpi": {
                        "thr_bps": thr_bps,
                        "avg_hol_ms": avg_hol_ms,
                        "outage_ratio": outage_ratio,
                        "avg_sinr_db": avg_sinr_db,
                        "queue_ratio": queue_ratio,
                        "drops_bits_win": drops_bits_win
                    },
                    "action": {"A_avg_scale": A_avg_scale, "B_mode": B_mode},
                    "history": {"A_avg_scale": history_A_scale, "acc_B": history_acc_B},
                    "dist": {"B_mode_counts": {str(k): v for k, v in b_mode_stats.items()}},
                }
                merged = _merge_dashboard(_read_json_safe(cfg.status_json), patch)
                _atomic_write_json(cfg.status_json, merged, retry=cfg.json_retry)

            # Terminal print
            if global_step % cfg.print_interval == 0:
                print(
                    f"Step {global_step} | nCUE={n_center_ues} | Mode={B_mode} | "
                    f"SINR={avg_sinr_db:.2f} dB | Outage={outage_ratio:.3f} | "
                    f"HOL={avg_hol_ms:.1f} ms | Thr={thr_bps/1e6:.3f} Mb/s | "
                    f"A_scale(avg)={A_avg_scale:.3f} | R={float(shaped_r):.3f}"
                )

            # advance
            obs = next_obs
            global_step += 1

            # periodic reset to avoid degenerate long episodes
            if global_step % cfg.reset_every_steps == 0:
                obs0, _ = env.reset(seed=cfg.reset_seed if cfg.fixed_reset_seed else None)
                obs = _to_vec(obs0)

            # PPO update when buffer full
            if buf.full():
                with torch.no_grad():
                    last_v = vf(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
                buf.compute_gae(last_v=last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)

                for _ in range(cfg.ppo_epochs):
                    for mb_idx in buf.get_minibatches(cfg.minibatch_size):
                        mb_obs = buf.obs[mb_idx]
                        mb_a2 = buf.a2[mb_idx]
                        mb_b = buf.b[mb_idx]
                        mb_old_logp_a = buf.logp_a[mb_idx]
                        mb_old_logp_b = buf.logp_b[mb_idx]
                        mb_adv = buf.adv[mb_idx]
                        mb_ret = buf.ret[mb_idx]

                        mu, std = pi_a(mb_obs)
                        logp_a = _squashed_gaussian_logp(mu, std, mb_a2)

                        logits_b = pi_b(mb_obs)
                        dist_b = torch.distributions.Categorical(logits=logits_b)
                        logp_b = dist_b.log_prob(mb_b)
                        ent_b = dist_b.entropy().mean()

                        ratio_a = torch.exp(logp_a - mb_old_logp_a)
                        ratio_b = torch.exp(logp_b - mb_old_logp_b)

                        loss_pi_a = -torch.mean(torch.min(
                            ratio_a * mb_adv,
                            torch.clamp(ratio_a, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                        ))
                        loss_pi_b = -torch.mean(torch.min(
                            ratio_b * mb_adv,
                            torch.clamp(ratio_b, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                        ))

                        v_pred = vf(mb_obs)
                        loss_v = torch.mean((v_pred - mb_ret) ** 2)

                        loss = loss_pi_a + loss_pi_b + cfg.vf_coef * loss_v - cfg.ent_coef * ent_b

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            list(pi_a.parameters()) + list(pi_b.parameters()) + list(vf.parameters()),
                            cfg.max_grad_norm
                        )
                        optimizer.step()

                buf.reset()


if __name__ == "__main__":
    main()
