"""
WM-only agent experiment: world model as the sole source of behavior.

No trained policies. Both agents are greedy planners: evaluate all action
sequences of length H through their WM, pick the best first action.

Three conditions (symmetric — no quality confound):
  Aligned-AA:   agent 0 = WM-AA, agent 1 = WM-AA
  Aligned-BB:   agent 0 = WM-BB, agent 1 = WM-BB
  Misaligned:   agent 0 = WM-AA, agent 1 = WM-BB

Each agent in Misaligned uses the WM trained on its own ego distribution
(AA for agent 0, BB for agent 1), so neither agent has a model-quality
advantage. If Misaligned still outperforms both Aligned conditions, the
effect is due to convention divergence, not quality.

WM-AA and WM-BB are trained in run_crossplay_identification.py.

Horizon sweep: H=1 (5 seqs), H=2 (25 seqs), H=3 (125 seqs) — exhaustive,
no sampling noise.

Landmark assignment logging: at episode end, record which landmark each
agent is nearest. "Split" = agents on different landmarks (correct coverage).
"Same" = both nearest the same landmark (failed symmetry breaking).
The symmetry-breaking hypothesis predicts: Aligned > Same, Misaligned > Split.
"""

import argparse
import json
import numpy as np
from itertools import product
from pathlib import Path
from scipy import stats

from train_world_models import load_world_model
from coord_env import (
    make_env, parse_obs, simultaneous_coverage_reward,
    ACT_DIM, MAX_CYCLES
)


def predicted_reward(wm, obs_self, obs_partner, action, lm_abs):
    """One-step: predict next obs, extract positions, compute reward."""
    next_obs  = wm.predict_np(obs_self, obs_partner, action)
    pos_self  = next_obs[2:4]
    pos_other = pos_self + next_obs[8:10]
    return simultaneous_coverage_reward(pos_self, pos_other, lm_abs)


def wm_greedy_action(wm, obs_self, obs_partner, lm_abs, horizon=1):
    """
    Pick the best first action exhaustively over all H-step sequences.
    H=1: 5 evals. H=2: 25. H=3: 125.
    """
    if horizon == 1:
        scores = [predicted_reward(wm, obs_self, obs_partner, a, lm_abs)
                  for a in range(ACT_DIM)]
        return int(np.argmax(scores))

    best_action, best_score = 0, -float('inf')
    for seq in product(range(ACT_DIM), repeat=horizon):
        total = 0.0
        cur_self    = obs_self.copy()
        cur_partner = obs_partner.copy()
        for a in seq:
            next_self    = wm.predict_np(cur_self, cur_partner, a)
            next_partner = wm.predict_np(cur_partner, cur_self, 0)
            pos_self  = next_self[2:4]
            pos_other = pos_self + next_self[8:10]
            total += simultaneous_coverage_reward(pos_self, pos_other, lm_abs)
            cur_self    = next_self
            cur_partner = next_partner
        if total > best_score:
            best_score  = total
            best_action = seq[0]
    return int(best_action)


def run_episode(wm_0, wm_1, seed, horizon=1):
    """
    Run one episode; both agents plan greedily with their respective WMs.
    Returns (total_return, landmark_assignment) where assignment is
    (lm_idx_agent0, lm_idx_agent1) at episode end, or None if invalid.
    """
    env = make_env()
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    if len(agents) < 2:
        env.close()
        return 0.0, None

    agent0, agent1 = agents[0], agents[1]
    total    = 0.0
    last_obs = {a: obs[a].copy() for a in [agent0, agent1]}

    for _ in range(MAX_CYCLES):
        if not env.agents:
            break

        actions = {}
        if agent0 in env.agents:
            _, _, lm_abs, _ = parse_obs(obs[agent0])
            actions[agent0] = wm_greedy_action(wm_0, obs[agent0], obs[agent1], lm_abs, horizon)
        if agent1 in env.agents:
            _, _, lm_abs, _ = parse_obs(obs[agent1])
            actions[agent1] = wm_greedy_action(wm_1, obs[agent1], obs[agent0], lm_abs, horizon)

        last_obs = {a: obs[a].copy() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        if rewards:
            total += sum(rewards.values()) / len(rewards)

    # Final landmark assignment
    assignment = None
    if agent0 in last_obs and agent1 in last_obs:
        _, pos0, lm_abs0, _ = parse_obs(last_obs[agent0])
        _, pos1, lm_abs1, _ = parse_obs(last_obs[agent1])
        lm0 = int(np.argmin([np.linalg.norm(pos0 - lm) for lm in lm_abs0]))
        lm1 = int(np.argmin([np.linalg.norm(pos1 - lm) for lm in lm_abs1]))
        assignment = (lm0, lm1)

    env.close()
    return total, assignment


def run_condition(name, wm_0, wm_1, n_episodes, horizon, seed_offset=0):
    returns, assignments = [], []
    for ep in range(n_episodes):
        ret, assign = run_episode(wm_0, wm_1, seed=seed_offset + ep, horizon=horizon)
        returns.append(ret)
        if assign is not None:
            assignments.append(assign)

    mean, std = np.mean(returns), np.std(returns)
    same  = sum(1 for a0, a1 in assignments if a0 == a1)
    split = len(assignments) - same
    split_pct = 100 * split / len(assignments) if assignments else 0
    print(f"    {name}: {mean:.3f} ± {std:.3f}  |  split={split_pct:.0f}%  same={100-split_pct:.0f}%",
          flush=True)
    return returns, {"same": same, "split": split, "split_pct": split_pct}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--horizons",   type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("WM-ONLY AGENT: Symmetric 3-condition alignment experiment")
    print(f"n_episodes={args.n_episodes}, horizons={args.horizons}")
    print("=" * 60)

    wm_AA = load_world_model("outputs/world_models/wm_AA.pt", device=args.device)
    wm_BB = load_world_model("outputs/world_models/wm_BB.pt", device=args.device)

    all_results = {}

    for horizon in args.horizons:
        print(f"\n{'='*60}")
        print(f"HORIZON = {horizon}")
        print(f"{'='*60}")

        ret_AA, assign_AA = run_condition(
            "Aligned-AA  (AA, AA)", wm_AA, wm_AA,
            args.n_episodes, horizon, seed_offset=0
        )
        ret_BB, assign_BB = run_condition(
            "Aligned-BB  (BB, BB)", wm_BB, wm_BB,
            args.n_episodes, horizon, seed_offset=10000
        )
        ret_mis, assign_mis = run_condition(
            "Misaligned  (AA, BB)", wm_AA, wm_BB,
            args.n_episodes, horizon, seed_offset=20000
        )

        t_AA, p_AA = stats.ttest_ind(ret_AA, ret_mis)
        t_BB, p_BB = stats.ttest_ind(ret_BB, ret_mis)
        gap_AA = np.mean(ret_AA) - np.mean(ret_mis)
        gap_BB = np.mean(ret_BB) - np.mean(ret_mis)

        print(f"\n    Aligned-AA vs Misaligned: gap={gap_AA:.3f}  p={p_AA:.4f}"
              + (" *" if p_AA < 0.05 else ""))
        print(f"    Aligned-BB vs Misaligned: gap={gap_BB:.3f}  p={p_BB:.4f}"
              + (" *" if p_BB < 0.05 else ""))
        if gap_AA < 0 and gap_BB < 0:
            print("    *** Misaligned outperforms BOTH aligned conditions ***")

        all_results[horizon] = {
            "aligned_AA":  {"mean": float(np.mean(ret_AA)), "std": float(np.std(ret_AA)),
                            "assignments": assign_AA},
            "aligned_BB":  {"mean": float(np.mean(ret_BB)), "std": float(np.std(ret_BB)),
                            "assignments": assign_BB},
            "misaligned":  {"mean": float(np.mean(ret_mis)), "std": float(np.std(ret_mis)),
                            "assignments": assign_mis},
            "gap_AA_vs_mis": float(gap_AA), "p_AA_vs_mis": float(p_AA),
            "gap_BB_vs_mis": float(gap_BB), "p_BB_vs_mis": float(p_BB),
        }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"{'H':>3} | {'Aligned-AA':>10} | {'Aligned-BB':>10} | {'Misaligned':>10} | "
          f"{'Gap-AA':>8} | {'p-AA':>7} | {'Gap-BB':>8} | {'p-BB':>7}")
    print("-" * 80)
    for h, r in all_results.items():
        print(f"{h:>3} | {r['aligned_AA']['mean']:>10.3f} | {r['aligned_BB']['mean']:>10.3f} | "
              f"{r['misaligned']['mean']:>10.3f} | {r['gap_AA_vs_mis']:>8.3f} | "
              f"{r['p_AA_vs_mis']:>7.4f} | {r['gap_BB_vs_mis']:>8.3f} | {r['p_BB_vs_mis']:>7.4f}")
    print("=" * 60)

    print("\nLandmark split rates (split = agents on different landmarks):")
    print(f"{'H':>3} | {'Aligned-AA split%':>18} | {'Aligned-BB split%':>18} | {'Misaligned split%':>18}")
    print("-" * 65)
    for h, r in all_results.items():
        print(f"{h:>3} | {r['aligned_AA']['assignments']['split_pct']:>18.1f} | "
              f"{r['aligned_BB']['assignments']['split_pct']:>18.1f} | "
              f"{r['misaligned']['assignments']['split_pct']:>18.1f}")
    print("=" * 65)

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wm_only_agent_results.json", "w") as f:
        json.dump({str(h): v for h, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to outputs/results/wm_only_agent_results.json")
