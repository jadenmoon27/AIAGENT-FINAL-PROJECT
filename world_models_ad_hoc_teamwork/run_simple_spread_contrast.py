"""
Simple spread contrast experiment.

Runs all three diagnostic metrics on vanilla simple_spread (N=2, no momentum
coupling, independent per-landmark coverage reward). Conventions should not form
because each agent's reward and transitions are independent of the partner.

Expected results:
  Metric 1 — Convention Divergence Score: near 1.0x (no cross-pop divergence)
  Metric 2 — Partner Identification Accuracy: ~50% at all K
  Metric 3 — Planning Impact: no significant gap between aligned/misaligned

This serves as a contrast case showing the protocol discriminates between
coupled (coord_env) and decoupled (simple_spread) settings.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from pathlib import Path
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset, random_split

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Simple spread env wrapper
# ---------------------------------------------------------------------------
SS_OBS_DIM = 12   # 2-agent simple_spread: vel(2)+pos(2)+lm_rel(4)+other_rel(4)
SS_ACT_DIM = 5
SS_MAX_CYCLES = 25


def make_ss_env():
    from pettingzoo.mpe import simple_spread_v3
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = simple_spread_v3.parallel_env(N=2, max_cycles=SS_MAX_CYCLES,
                                             continuous_actions=False,
                                             local_ratio=0.5)
    return env


# ---------------------------------------------------------------------------
# World model (same architecture, adapted for SS_OBS_DIM)
# ---------------------------------------------------------------------------
SS_IN_DIM  = SS_OBS_DIM * 2 + SS_ACT_DIM
SS_OUT_DIM = SS_OBS_DIM


class SSWorldModel(nn.Module):
    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        layers = []
        in_d = SS_IN_DIM
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, SS_OUT_DIM))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean",  torch.zeros(SS_IN_DIM))
        self.register_buffer("input_std",   torch.ones(SS_IN_DIM))
        self.register_buffer("output_mean", torch.zeros(SS_OUT_DIM))
        self.register_buffer("output_std",  torch.ones(SS_OUT_DIM))

    def forward(self, x):
        xn = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(xn) * (self.output_std + 1e-8) + self.output_mean

    def predict_np(self, obs_self, obs_partner, action):
        act = np.zeros(SS_ACT_DIM, dtype=np.float32)
        act[action] = 1.0
        x = np.concatenate([obs_self, obs_partner, act])
        with torch.no_grad():
            xt = torch.FloatTensor(x).unsqueeze(0)
            return self.forward(xt).squeeze(0).cpu().numpy()


def train_ss_wm(obs_self, obs_partner, actions, next_obs, epochs=100):
    model = SSWorldModel()
    N = len(actions)
    act_oh = np.zeros((N, SS_ACT_DIM), dtype=np.float32)
    act_oh[np.arange(N), actions] = 1.0
    X = np.concatenate([obs_self, obs_partner, act_oh], axis=1).astype(np.float32)
    Y = next_obs.astype(np.float32)

    model.input_mean  = torch.FloatTensor(X.mean(0))
    model.input_std   = torch.FloatTensor(X.std(0))
    model.output_mean = torch.FloatTensor(Y.mean(0))
    model.output_std  = torch.FloatTensor(Y.std(0))

    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    val_n = max(1, int(len(ds) * 0.1))
    tr_ds, va_ds = random_split(ds, [len(ds) - val_n, val_n])
    tr_loader = DataLoader(tr_ds, batch_size=512, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=512)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in va_loader) / val_n
        if vl < best:
            best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 15: break

    model.load_state_dict(best_state)
    return model.eval(), best


# ---------------------------------------------------------------------------
# Policy training (simple PPO wrapper)
# ---------------------------------------------------------------------------
class RandomPolicy:
    """Random policy — no conventions, maximum decoupling."""
    def __init__(self, act_dim, seed=0):
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):
        return self.rng.integers(0, self.act_dim), None


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------
def collect_ss_trajectories(policy_ego, policy_partner, n_episodes, seed_offset=0):
    obs_self_l, obs_part_l, act_l, next_obs_l = [], [], [], []
    env = make_ss_env()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        agents = list(env.agents)
        if len(agents) < 2: continue
        a0, a1 = agents[0], agents[1]
        for _ in range(SS_MAX_CYCLES):
            if not env.agents: break
            actions = {}
            if a0 in env.agents:
                act, _ = policy_ego.predict(obs[a0], deterministic=False)
                actions[a0] = int(act)
            if a1 in env.agents:
                act, _ = policy_partner.predict(obs[a1], deterministic=False)
                actions[a1] = int(act)
            prev = {a: obs[a].copy() for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
            if a0 in obs and a1 in prev:
                obs_self_l.append(prev[a0])
                obs_part_l.append(prev[a1])
                act_l.append(actions.get(a0, 0))
                next_obs_l.append(obs[a0])
    env.close()
    return (np.array(obs_self_l, dtype=np.float32),
            np.array(obs_part_l, dtype=np.float32),
            np.array(act_l, dtype=np.int64),
            np.array(next_obs_l, dtype=np.float32))


# ---------------------------------------------------------------------------
# Metric 1: Convention Divergence Score
# ---------------------------------------------------------------------------
def metric1_divergence(wm_AA, wm_BB, wm_AA2, wm_BB2, n_samples=2000, seed=0):
    """
    Cross-pop divergence (wm_AA vs wm_BB) vs self-divergence (wm_AA vs wm_AA2).
    Both measured on the same random inputs.
    """
    rng = np.random.default_rng(seed)
    obs_s = rng.uniform(-1, 1, (n_samples, SS_OBS_DIM)).astype(np.float32)
    obs_p = rng.uniform(-1, 1, (n_samples, SS_OBS_DIM)).astype(np.float32)
    acts  = rng.integers(0, SS_ACT_DIM, n_samples)

    cross_divs, self_divs = [], []
    for i in range(n_samples):
        p_AA  = wm_AA.predict_np(obs_s[i], obs_p[i], int(acts[i]))
        p_BB  = wm_BB.predict_np(obs_s[i], obs_p[i], int(acts[i]))
        p_AA2 = wm_AA2.predict_np(obs_s[i], obs_p[i], int(acts[i]))
        cross_divs.append(np.linalg.norm(p_AA - p_BB))
        self_divs.append(np.linalg.norm(p_AA - p_AA2))

    cross_mean = np.mean(cross_divs)
    self_mean  = np.mean(self_divs)
    ratio = cross_mean / (self_mean + 1e-10)
    return cross_mean, self_mean, ratio


# ---------------------------------------------------------------------------
# Metric 2: Partner Identification Accuracy
# ---------------------------------------------------------------------------
def metric2_identification(policy_A, policy_B, wm_AA, wm_AB, n_episodes=500,
                            k_values=None, error_slice=None):
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10, 15, 20]
    results = {k: {"correct": 0, "total": 0} for k in k_values}

    for ep in range(n_episodes):
        use_A = (ep % 2 == 0)
        partner = policy_A if use_A else policy_B
        true_pop = "A" if use_A else "B"

        env = make_ss_env()
        obs, _ = env.reset(seed=ep)
        agents = list(env.agents)
        if len(agents) < 2:
            env.close(); continue
        a0, a1 = agents[0], agents[1]
        err_AA = err_AB = 0.0
        step = 0

        for _ in range(SS_MAX_CYCLES):
            if not env.agents: break
            actions = {}
            if a0 in env.agents:
                act, _ = policy_A.predict(obs[a0], deterministic=False)
                actions[a0] = int(act)
            if a1 in env.agents:
                act, _ = partner.predict(obs[a1], deterministic=False)
                actions[a1] = int(act)
            prev = {a: obs[a].copy() for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
            step += 1
            if a0 not in obs or a1 not in prev: break

            pred_AA = wm_AA.predict_np(prev[a0], prev[a1], actions.get(a0, 0))
            pred_AB = wm_AB.predict_np(prev[a0], prev[a1], actions.get(a0, 0))
            actual  = obs[a0]

            if error_slice is not None:
                err_AA += np.linalg.norm(pred_AA[error_slice] - actual[error_slice])
                err_AB += np.linalg.norm(pred_AB[error_slice] - actual[error_slice])
            else:
                err_AA += np.linalg.norm(pred_AA - actual)
                err_AB += np.linalg.norm(pred_AB - actual)

            if step in k_values:
                predicted = "A" if err_AA <= err_AB else "B"
                results[step]["correct"] += int(predicted == true_pop)
                results[step]["total"]   += 1

        env.close()
    return results


# ---------------------------------------------------------------------------
# Metric 3: Planning Impact
# ---------------------------------------------------------------------------
def ss_greedy_action(wm, obs_self, obs_partner, horizon=1):
    if horizon == 1:
        scores = []
        for a in range(SS_ACT_DIM):
            next_obs = wm.predict_np(obs_self, obs_partner, a)
            # Simple spread reward proxy: negative distance to mean landmark position
            # (encoded in obs[4:8] as landmark relatives)
            lm_rels = next_obs[4:8].reshape(2, 2)
            pos_self = next_obs[2:4]
            lm_abs = pos_self + lm_rels
            dists = [np.linalg.norm(lm_rels[i]) for i in range(2)]
            scores.append(-min(dists))  # reward = negative min distance to any landmark
        return int(np.argmax(scores))

    best_action, best_score = 0, -float('inf')
    for seq in product(range(SS_ACT_DIM), repeat=horizon):
        total = 0.0
        cur_s, cur_p = obs_self.copy(), obs_partner.copy()
        for a in seq:
            next_s = wm.predict_np(cur_s, cur_p, a)
            next_p = wm.predict_np(cur_p, cur_s, 0)
            lm_rels = next_s[4:8].reshape(2, 2)
            dists = [np.linalg.norm(lm_rels[i]) for i in range(2)]
            total += -min(dists)
            cur_s, cur_p = next_s, next_p
        if total > best_score:
            best_score = total; best_action = seq[0]
    return int(best_action)


def run_ss_episode(wm_0, wm_1, seed, horizon=1):
    env = make_ss_env()
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    if len(agents) < 2:
        env.close(); return 0.0
    a0, a1 = agents[0], agents[1]
    total = 0.0
    for _ in range(SS_MAX_CYCLES):
        if not env.agents: break
        actions = {}
        if a0 in env.agents:
            actions[a0] = ss_greedy_action(wm_0, obs[a0], obs[a1], horizon)
        if a1 in env.agents:
            actions[a1] = ss_greedy_action(wm_1, obs[a1], obs[a0], horizon)
        obs, rewards, _, _, _ = env.step(actions)
        if rewards:
            total += sum(rewards.values()) / len(rewards)
    env.close()
    return total


def run_ss_condition(name, wm_0, wm_1, n_episodes, horizon, seed_offset=0):
    returns = [run_ss_episode(wm_0, wm_1, seed_offset + ep, horizon)
               for ep in range(n_episodes)]
    mean, std = np.mean(returns), np.std(returns)
    print(f"    {name}: {mean:.3f} ± {std:.3f}", flush=True)
    return returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",    type=int, default=300)
    parser.add_argument("--traj-episodes", type=int, default=2000)
    parser.add_argument("--wm-epochs",     type=int, default=100)
    parser.add_argument("--horizons",      type=int, nargs="+", default=[1, 2])
    parser.add_argument("--skip-train",    action="store_true")
    args = parser.parse_args()

    out_dir = Path("outputs/simple_spread")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIMPLE SPREAD CONTRAST — 3-Metric Protocol")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Use random policies — no conventions by design
    # This is the correct choice for the contrast case: simple_spread with
    # random agents has no convention structure to encode.
    # ------------------------------------------------------------------
    print("\nUsing random policies (no convention structure by design).")
    policy_A  = RandomPolicy(SS_ACT_DIM, seed=0)
    policy_B  = RandomPolicy(SS_ACT_DIM, seed=42)
    policy_A2 = RandomPolicy(SS_ACT_DIM, seed=1)

    # ------------------------------------------------------------------
    # Collect trajectories and train WMs
    # ------------------------------------------------------------------
    wm_AA_path  = out_dir / "wm_ss_AA.pt"
    wm_AB_path  = out_dir / "wm_ss_AB.pt"
    wm_BB_path  = out_dir / "wm_ss_BB.pt"
    wm_AA2_path = out_dir / "wm_ss_AA2.pt"

    if not args.skip_train:
        print(f"\nCollecting trajectories ({args.traj_episodes} eps each)...")
        os_AA, op_AA, ac_AA, no_AA = collect_ss_trajectories(policy_A,  policy_A,  args.traj_episodes, 0)
        os_AB, op_AB, ac_AB, no_AB = collect_ss_trajectories(policy_A,  policy_B,  args.traj_episodes, 100000)
        os_BB, op_BB, ac_BB, no_BB = collect_ss_trajectories(policy_B,  policy_B,  args.traj_episodes, 200000)
        os_A2, op_A2, ac_A2, no_A2 = collect_ss_trajectories(policy_A2, policy_A2, args.traj_episodes, 300000)
        print(f"  AA: {len(ac_AA)}, AB: {len(ac_AB)}, BB: {len(ac_BB)}, A2A2: {len(ac_A2)} transitions")

        print("\nTraining WM-AA...")
        wm_AA,  vl_AA  = train_ss_wm(os_AA, op_AA, ac_AA, no_AA, args.wm_epochs)
        print(f"  val loss: {vl_AA:.6f}")
        torch.save({"state_dict": wm_AA.state_dict()},  str(wm_AA_path))

        print("Training WM-AB...")
        wm_AB,  vl_AB  = train_ss_wm(os_AB, op_AB, ac_AB, no_AB, args.wm_epochs)
        print(f"  val loss: {vl_AB:.6f}")
        torch.save({"state_dict": wm_AB.state_dict()},  str(wm_AB_path))

        print("Training WM-BB...")
        wm_BB,  vl_BB  = train_ss_wm(os_BB, op_BB, ac_BB, no_BB, args.wm_epochs)
        print(f"  val loss: {vl_BB:.6f}")
        torch.save({"state_dict": wm_BB.state_dict()},  str(wm_BB_path))

        print("Training WM-AA2 (self-divergence baseline)...")
        wm_AA2, vl_AA2 = train_ss_wm(os_A2, op_A2, ac_A2, no_A2, args.wm_epochs)
        print(f"  val loss: {vl_AA2:.6f}")
        torch.save({"state_dict": wm_AA2.state_dict()}, str(wm_AA2_path))
    else:
        def load(p):
            m = SSWorldModel()
            m.load_state_dict(torch.load(str(p), map_location="cpu", weights_only=False)["state_dict"])
            return m.eval()
        wm_AA = load(wm_AA_path); wm_AB = load(wm_AB_path)
        wm_BB = load(wm_BB_path); wm_AA2 = load(wm_AA2_path)

    # ------------------------------------------------------------------
    # Metric 1: Convention Divergence Score
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("METRIC 1 — Convention Divergence Score")
    print("=" * 60)
    cross_mean, self_mean, ratio = metric1_divergence(wm_AA, wm_BB, wm_AA2, wm_BB)
    print(f"  Cross-pop divergence (AA vs BB): {cross_mean:.6f}")
    print(f"  Self-divergence      (AA vs AA2): {self_mean:.6f}")
    print(f"  Ratio (cross / self): {ratio:.2f}x")
    print(f"  coord_env_v2 ratio for reference: ~152x")
    if ratio < 5:
        print("  => Near noise floor. No convention encoding detected.")
    else:
        print("  => Convention encoding present.")

    # ------------------------------------------------------------------
    # Metric 2: Partner Identification
    # ------------------------------------------------------------------
    k_values = [1, 2, 3, 5, 8, 10, 15, 20]
    print("\n" + "=" * 60)
    print("METRIC 2 — Partner Identification Accuracy")
    print("=" * 60)

    res_full = metric2_identification(policy_A, policy_B, wm_AA, wm_AB,
                                      n_episodes=args.n_episodes, k_values=k_values)
    res_rel  = metric2_identification(policy_A, policy_B, wm_AA, wm_AB,
                                      n_episodes=args.n_episodes, k_values=k_values,
                                      error_slice=slice(8, 12))

    print(f"\n  {'K':>4} | {'Full obs':>10} | {'other_rel':>10}")
    print(f"  {'-'*30}")
    for k in k_values:
        acc_f = res_full[k]["correct"] / res_full[k]["total"] if res_full[k]["total"] > 0 else 0
        acc_r = res_rel[k]["correct"]  / res_rel[k]["total"]  if res_rel[k]["total"]  > 0 else 0
        print(f"  {k:>4} | {acc_f:>10.3f} | {acc_r:>10.3f}")
    print(f"  Random baseline: 0.500")

    # ------------------------------------------------------------------
    # Metric 3: Planning Impact
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("METRIC 3 — Planning Impact (horizon sweep)")
    print("=" * 60)

    planning_results = {}
    for h in args.horizons:
        print(f"\n  HORIZON = {h}")
        ret_AA  = run_ss_condition("Aligned-AA  (AA,AA)", wm_AA, wm_AA, args.n_episodes, h, 0)
        ret_BB  = run_ss_condition("Aligned-BB  (BB,BB)", wm_BB, wm_BB, args.n_episodes, h, 10000)
        ret_mis = run_ss_condition("Misaligned  (AA,BB)", wm_AA, wm_BB, args.n_episodes, h, 20000)
        _, p_AA = stats.ttest_ind(ret_AA, ret_mis)
        _, p_BB = stats.ttest_ind(ret_BB, ret_mis)
        gap_AA = np.mean(ret_AA) - np.mean(ret_mis)
        gap_BB = np.mean(ret_BB) - np.mean(ret_mis)
        print(f"    AA vs Mis: gap={gap_AA:.3f}  p={p_AA:.4f}")
        print(f"    BB vs Mis: gap={gap_BB:.3f}  p={p_BB:.4f}")
        planning_results[h] = {
            "aligned_AA": float(np.mean(ret_AA)), "aligned_BB": float(np.mean(ret_BB)),
            "misaligned": float(np.mean(ret_mis)),
            "gap_AA": float(gap_AA), "p_AA": float(p_AA),
            "gap_BB": float(gap_BB), "p_BB": float(p_BB),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "metric1": {"cross_divergence": cross_mean, "self_divergence": self_mean, "ratio": ratio},
        "metric2_full": {str(k): v for k, v in res_full.items()},
        "metric2_rel":  {str(k): v for k, v in res_rel.items()},
        "metric3": {str(h): v for h, v in planning_results.items()},
    }
    with open(out_dir / "simple_spread_contrast_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to outputs/simple_spread/simple_spread_contrast_results.json")
