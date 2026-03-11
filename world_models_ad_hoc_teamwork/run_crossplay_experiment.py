"""
Cross-play experiment: does convention mismatch hurt performance?

Conditions:
  Self-play A:   Pop-A agent0 + Pop-A agent1 (matched conventions)
  Self-play B:   Pop-B agent0 + Pop-B agent1 (matched conventions)
  Cross-play AB: Pop-A agent0 + Pop-B agent1 (mismatched conventions)
  Cross-play BA: Pop-B agent0 + Pop-A agent1 (mismatched conventions)
  Adaptive-K:    agent0 observes K steps, identifies partner via WM prediction
                 error, switches to matched policy for remaining steps.
                 Swept over K = 0, 1, 2, 3, 5, 8, 10.

The WM is used ONLY for inference (single-step prediction error against real
observations) — no autoregressive rollout, no compounding error.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from stable_baselines3 import PPO

from train_world_models import load_world_model
from coord_env import make_env, MAX_CYCLES


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_selfplay(policy, n_episodes, seed_offset=0):
    """Both agents use the same policy."""
    returns = []
    for ep in range(n_episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed_offset + ep)
        total = 0.0
        for _ in range(MAX_CYCLES):
            if not env.agents:
                break
            actions = {}
            for agent in env.agents:
                act, _ = policy.predict(obs[agent], deterministic=False)
                actions[agent] = int(act)
            obs, rewards, _, _, _ = env.step(actions)
            total += sum(rewards.values()) / len(rewards)
        env.close()
        returns.append(total)
    return returns


def run_crossplay(policy_0, policy_1, n_episodes, seed_offset=0):
    """Agent 0 uses policy_0, agent 1 uses policy_1."""
    returns = []
    for ep in range(n_episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed_offset + ep)
        total = 0.0
        agents_fixed = None
        for _ in range(MAX_CYCLES):
            if not env.agents:
                break
            if agents_fixed is None:
                agents_fixed = list(env.agents)
            actions = {}
            for agent in env.agents:
                policy = policy_0 if agent == agents_fixed[0] else policy_1
                act, _ = policy.predict(obs[agent], deterministic=False)
                actions[agent] = int(act)
            obs, rewards, _, _, _ = env.step(actions)
            total += sum(rewards.values()) / len(rewards)
        env.close()
        returns.append(total)
    return returns


def run_adaptive(policy_A, policy_B, wm_A, wm_B,
                 true_partner_policy,  # the actual partner (A or B)
                 true_partner_pop,     # "A" or "B" — for oracle comparison
                 k_observe, n_episodes, seed_offset=0):
    """
    Agent 0 observes for k_observe steps, accumulates WM prediction error,
    then commits to the matched policy for the rest of the episode.
    Agent 1 always uses true_partner_policy.
    """
    returns = []
    correct_ids = 0

    for ep in range(n_episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed_offset + ep)
        total = 0.0

        agents_fixed = None
        err_A, err_B = 0.0, 0.0
        chosen_policy = None  # committed after k_observe steps
        step = 0

        for _ in range(MAX_CYCLES):
            if not env.agents:
                break
            if agents_fixed is None:
                agents_fixed = list(env.agents)

            agent0 = agents_fixed[0]
            agent1 = agents_fixed[1] if len(agents_fixed) > 1 else None

            actions = {}

            # Agent 0: use policy_A during observation phase, then committed policy
            if agent0 in env.agents:
                if chosen_policy is None:
                    # Observation phase: use policy_A as default
                    act, _ = policy_A.predict(obs[agent0], deterministic=False)
                else:
                    act, _ = chosen_policy.predict(obs[agent0], deterministic=False)
                actions[agent0] = int(act)

            # Agent 1: always true partner policy
            if agent1 and agent1 in env.agents:
                act, _ = true_partner_policy.predict(obs[agent1], deterministic=False)
                actions[agent1] = int(act)

            prev_obs = {a: obs[a].copy() for a in env.agents}
            obs, rewards, _, _, _ = env.step(actions)
            total += sum(rewards.values()) / len(rewards)
            step += 1

            # Accumulate WM prediction error during observation phase
            if chosen_policy is None and agent0 in obs and agent1 and agent1 in prev_obs:
                pred_A = wm_A.predict_np(prev_obs[agent0], prev_obs[agent1], actions[agent0])
                pred_B = wm_B.predict_np(prev_obs[agent0], prev_obs[agent1], actions[agent0])
                actual = obs[agent0]
                err_A += np.linalg.norm(pred_A - actual)
                err_B += np.linalg.norm(pred_B - actual)

                if step >= k_observe:
                    # Commit to matched policy
                    predicted_pop = "A" if err_A <= err_B else "B"
                    chosen_policy = policy_A if predicted_pop == "A" else policy_B
                    if predicted_pop == true_partner_pop:
                        correct_ids += 1

        env.close()
        returns.append(total)

    id_acc = correct_ids / n_episodes if k_observe > 0 else None
    return returns, id_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    print("="*60)
    print("CROSS-PLAY EXPERIMENT")
    print(f"n_episodes={args.n_episodes}")
    print("="*60)

    policy_A = PPO.load("outputs/policies/policy_A")
    policy_B = PPO.load("outputs/policies/policy_B")
    wm_A = load_world_model("outputs/world_models/wm_A.pt", device=args.device)
    wm_B = load_world_model("outputs/world_models/wm_B.pt", device=args.device)

    results = {}

    # ------------------------------------------------------------------
    # Self-play baselines
    # ------------------------------------------------------------------
    print("\n--- Self-play ---")
    sp_A = run_selfplay(policy_A, args.n_episodes, seed_offset=0)
    print(f"  Self-play A:  {np.mean(sp_A):.3f} ± {np.std(sp_A):.3f}")
    sp_B = run_selfplay(policy_B, args.n_episodes, seed_offset=10000)
    print(f"  Self-play B:  {np.mean(sp_B):.3f} ± {np.std(sp_B):.3f}")
    results["selfplay_A"] = sp_A
    results["selfplay_B"] = sp_B

    # ------------------------------------------------------------------
    # Cross-play
    # ------------------------------------------------------------------
    print("\n--- Cross-play ---")
    cp_AB = run_crossplay(policy_A, policy_B, args.n_episodes, seed_offset=20000)
    print(f"  Cross-play AB: {np.mean(cp_AB):.3f} ± {np.std(cp_AB):.3f}")
    cp_BA = run_crossplay(policy_B, policy_A, args.n_episodes, seed_offset=30000)
    print(f"  Cross-play BA: {np.mean(cp_BA):.3f} ± {np.std(cp_BA):.3f}")
    results["crossplay_AB"] = cp_AB
    results["crossplay_BA"] = cp_BA

    self_play  = sp_A + sp_B
    cross_play = cp_AB + cp_BA
    t_cp, p_cp = stats.ttest_ind(self_play, cross_play)
    print(f"\n  Self-play mean:  {np.mean(self_play):.3f}")
    print(f"  Cross-play mean: {np.mean(cross_play):.3f}")
    print(f"  Gap: {np.mean(self_play) - np.mean(cross_play):.3f}")
    print(f"  t={t_cp:.3f}, p={p_cp:.4f}")

    if p_cp > 0.05:
        print("  WARNING: No significant cross-play gap — conventions may not diverge enough")
    else:
        print("  PASS: Significant cross-play gap — conventions diverge and cause coordination cost")

    # ------------------------------------------------------------------
    # Adaptive: sweep K
    # ------------------------------------------------------------------
    print("\n--- Adaptive (WM identification → policy switch) ---")
    k_values = [0, 1, 2, 3, 5, 8, 10]
    adaptive_results = []

    for k in k_values:
        # Test with Pop-B partner (harder case — agent 0 starts with wrong policy)
        ret_B, acc_B = run_adaptive(
            policy_A, policy_B, wm_A, wm_B,
            true_partner_policy=policy_B, true_partner_pop="B",
            k_observe=k, n_episodes=args.n_episodes, seed_offset=40000
        )
        # Test with Pop-A partner (easier case — agent 0 policy already matches)
        ret_A, acc_A = run_adaptive(
            policy_A, policy_B, wm_A, wm_B,
            true_partner_policy=policy_A, true_partner_pop="A",
            k_observe=k, n_episodes=args.n_episodes, seed_offset=50000
        )
        combined = ret_A + ret_B
        mean_acc = np.mean([a for a in [acc_A, acc_B] if a is not None])
        print(f"  K={k:>2}: return={np.mean(combined):.3f} ± {np.std(combined):.3f} | "
              f"id_acc={mean_acc:.3f}" if k > 0 else
              f"  K={k:>2}: return={np.mean(combined):.3f} ± {np.std(combined):.3f} | "
              f"id_acc=N/A (random)")
        adaptive_results.append({
            "k": k,
            "mean": float(np.mean(combined)),
            "std":  float(np.std(combined)),
            "id_acc": float(mean_acc) if k > 0 else None,
        })

    results["adaptive"] = adaptive_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Self-play:   {np.mean(self_play):.3f} ± {np.std(self_play):.3f}  (upper bound)")
    print(f"  Cross-play:  {np.mean(cross_play):.3f} ± {np.std(cross_play):.3f}  (lower bound)")
    print(f"  Gap:         {np.mean(self_play) - np.mean(cross_play):.3f}")
    print(f"  p={p_cp:.4f}")
    print()
    print(f"  {'K':>4} | {'Return':>8} | {'ID Acc':>8} | {'Gap recovered':>14}")
    print(f"  {'-'*42}")
    sp_mean = np.mean(self_play)
    cp_mean = np.mean(cross_play)
    gap_total = sp_mean - cp_mean
    for r in adaptive_results:
        recovered = (r['mean'] - cp_mean) / gap_total * 100 if gap_total > 0 else 0
        acc_str = f"{r['id_acc']:.3f}" if r['id_acc'] is not None else "  N/A"
        print(f"  {r['k']:>4} | {r['mean']:>8.3f} | {acc_str:>8} | {recovered:>13.1f}%")
    print("="*60)

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "crossplay_results.json", "w") as f:
        json.dump({
            "selfplay_mean":  float(np.mean(self_play)),
            "crossplay_mean": float(np.mean(cross_play)),
            "gap":            float(np.mean(self_play) - np.mean(cross_play)),
            "p":              float(p_cp),
            "adaptive":       adaptive_results,
        }, f, indent=2)
    print(f"\nSaved to outputs/results/crossplay_results.json")
