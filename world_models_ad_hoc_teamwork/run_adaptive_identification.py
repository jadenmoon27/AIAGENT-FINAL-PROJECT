"""
Pre-episode identification + policy switching experiment.

Closes the loop from identification to action:
  1. Warmup phase (K steps): ego agent uses a default policy while accumulating
     WM prediction error to identify partner convention.
  2. Switch phase: once identified, ego switches to the matching policy
     (policy_A if partner identified as Pop-A, policy_B if Pop-B) and plays
     out the remaining episode.

Baselines:
  - Always-A:    ego always uses policy_A (no adaptation)
  - Always-B:    ego always uses policy_B (no adaptation)
  - Oracle:      ego uses the ground-truth correct policy from step 1 (upper bound)
  - Adaptive-K:  ego uses identification for K steps, then switches

K sweep: [1, 3, 5, 10, 15] to show the identification-to-benefit curve.

Conditions:
  - Partner is Pop-A or Pop-B (50/50, alternating by episode)
  - Ego is always Pop-A by default (until switch)

Expected result:
  - Oracle > Adaptive-K > Always-A ≈ Always-B for partner-mixed episodes
  - Adaptive-K performance approaches Oracle as K increases (better ID → better switch)
  - Adaptive-K > Always-A at K=10 if identification (91.6%) is actionable
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from stable_baselines3 import PPO

from train_world_models import load_world_model
from coord_env import make_env, MAX_CYCLES, OBS_DIM

PARTNER_REL_SLICE = slice(8, 10)


def run_episode_adaptive(policy_ego_default, policy_ego_match,
                          policy_partner,
                          wm_AA, wm_AB,
                          seed, warmup_k,
                          error_slice=PARTNER_REL_SLICE):
    """
    Run one episode with K-step warmup identification then policy switch.
    Returns (total_return, identified_correctly).
    """
    env = make_env()
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    if len(agents) < 2:
        env.close()
        return 0.0, None

    a0, a1 = agents[0], agents[1]
    total = 0.0
    err_AA = err_AB = 0.0
    step = 0
    switched = False
    current_ego_policy = policy_ego_default

    for _ in range(MAX_CYCLES):
        if not env.agents: break

        actions = {}
        if a0 in env.agents:
            act, _ = current_ego_policy.predict(obs[a0], deterministic=False)
            actions[a0] = int(act)
        if a1 in env.agents:
            act, _ = policy_partner.predict(obs[a1], deterministic=False)
            actions[a1] = int(act)

        prev = {a: obs[a].copy() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        if rewards:
            total += sum(rewards.values()) / len(rewards)
        step += 1

        # Accumulate prediction error during warmup
        if not switched and a0 in obs and a1 in prev:
            pred_AA = wm_AA.predict_np(prev[a0], prev[a1], actions.get(a0, 0))
            pred_AB = wm_AB.predict_np(prev[a0], prev[a1], actions.get(a0, 0))
            actual  = obs[a0]
            err_AA += np.linalg.norm(pred_AA[error_slice] - actual[error_slice])
            err_AB += np.linalg.norm(pred_AB[error_slice] - actual[error_slice])

        # Switch after warmup_k steps
        if not switched and step >= warmup_k:
            identified_A = (err_AA <= err_AB)
            current_ego_policy = policy_ego_default if identified_A else policy_ego_match
            switched = True

    env.close()
    return total, (err_AA <= err_AB)


def run_episode_fixed(policy_ego, policy_partner, seed):
    """Run one episode with fixed ego policy (no switching)."""
    env = make_env()
    obs, _ = env.reset(seed=seed)
    agents = list(env.agents)
    if len(agents) < 2:
        env.close()
        return 0.0
    a0, a1 = agents[0], agents[1]
    total = 0.0
    for _ in range(MAX_CYCLES):
        if not env.agents: break
        actions = {}
        if a0 in env.agents:
            act, _ = policy_ego.predict(obs[a0], deterministic=False)
            actions[a0] = int(act)
        if a1 in env.agents:
            act, _ = policy_partner.predict(obs[a1], deterministic=False)
            actions[a1] = int(act)
        obs, rewards, _, _, _ = env.step(actions)
        if rewards:
            total += sum(rewards.values()) / len(rewards)
    env.close()
    return total


def run_condition(name, fn, n_episodes):
    returns = [fn(ep) for ep in range(n_episodes)]
    mean, std = np.mean(returns), np.std(returns)
    print(f"  {name:<30}: {mean:.3f} ± {std:.3f}", flush=True)
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--k-values",   type=int, nargs="+", default=[1, 3, 5, 10, 15])
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("ADAPTIVE IDENTIFICATION — Pre-episode warmup + policy switch")
    print(f"n_episodes={args.n_episodes}, K values={args.k_values}")
    print("=" * 60)

    policy_A = PPO.load("outputs/policies/policy_A")
    policy_B = PPO.load("outputs/policies/policy_B")
    wm_AA = load_world_model("outputs/world_models/wm_AA.pt", device=args.device)
    wm_AB = load_world_model("outputs/world_models/wm_AB.pt", device=args.device)

    n = args.n_episodes
    # Episodes alternate partner: even = Pop-A partner, odd = Pop-B partner
    seeds_A_partner = list(range(0, n, 2))       # Pop-A partner episodes
    seeds_B_partner = list(range(1, n, 2))       # Pop-B partner episodes

    all_results = {}

    print("\n--- Baselines ---")

    # Always-A: ego=policy_A, partner=A or B (mixed)
    def always_A(ep):
        partner = policy_A if ep % 2 == 0 else policy_B
        return run_episode_fixed(policy_A, partner, seed=ep)

    # Always-B: ego=policy_B, partner=A or B (mixed)
    def always_B(ep):
        partner = policy_A if ep % 2 == 0 else policy_B
        return run_episode_fixed(policy_B, partner, seed=ep)

    # Oracle: ego uses the correct matching policy from step 1
    def oracle(ep):
        partner = policy_A if ep % 2 == 0 else policy_B
        ego = policy_A if ep % 2 == 0 else policy_B
        return run_episode_fixed(ego, partner, seed=ep)

    ret_always_A = run_condition("Always-A (no adapt)", always_A, n)
    ret_always_B = run_condition("Always-B (no adapt)", always_B, n)
    ret_oracle   = run_condition("Oracle (perfect ID)",  oracle,   n)

    all_results["always_A"] = {"mean": float(np.mean(ret_always_A)), "std": float(np.std(ret_always_A))}
    all_results["always_B"] = {"mean": float(np.mean(ret_always_B)), "std": float(np.std(ret_always_B))}
    all_results["oracle"]   = {"mean": float(np.mean(ret_oracle)),   "std": float(np.std(ret_oracle))}

    print("\n--- Adaptive (K-step warmup then switch) ---")
    k_results = {}

    for k in args.k_values:
        def adaptive(ep, _k=k):
            partner = policy_A if ep % 2 == 0 else policy_B
            # policy_ego_match: switch to B if partner identified as B
            ret, _ = run_episode_adaptive(
                policy_ego_default=policy_A,
                policy_ego_match=policy_B,
                policy_partner=partner,
                wm_AA=wm_AA, wm_AB=wm_AB,
                seed=ep, warmup_k=_k,
            )
            return ret

        ret_adaptive = run_condition(f"Adaptive-K={k}", adaptive, n)

        _, p_vs_A = stats.ttest_ind(ret_adaptive, ret_always_A)
        _, p_vs_oracle = stats.ttest_ind(ret_adaptive, ret_oracle)
        gap_vs_A = np.mean(ret_adaptive) - np.mean(ret_always_A)
        gap_vs_oracle = np.mean(ret_adaptive) - np.mean(ret_oracle)

        print(f"    vs Always-A: gap={gap_vs_A:+.3f}  p={p_vs_A:.4f}"
              + (" *" if p_vs_A < 0.05 else ""))
        print(f"    vs Oracle:   gap={gap_vs_oracle:+.3f}  p={p_vs_oracle:.4f}")

        k_results[k] = {
            "mean": float(np.mean(ret_adaptive)),
            "std":  float(np.std(ret_adaptive)),
            "gap_vs_always_A": float(gap_vs_A),
            "p_vs_always_A":   float(p_vs_A),
            "gap_vs_oracle":   float(gap_vs_oracle),
            "p_vs_oracle":     float(p_vs_oracle),
        }

    all_results["adaptive"] = k_results

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  {'Condition':<25} | {'Mean':>8} | {'Std':>7}")
    print(f"  {'-'*45}")
    print(f"  {'Always-A':<25} | {all_results['always_A']['mean']:>8.3f} | {all_results['always_A']['std']:>7.3f}")
    print(f"  {'Always-B':<25} | {all_results['always_B']['mean']:>8.3f} | {all_results['always_B']['std']:>7.3f}")
    print(f"  {'Oracle':<25} | {all_results['oracle']['mean']:>8.3f} | {all_results['oracle']['std']:>7.3f}")
    for k, r in k_results.items():
        print(f"  {f'Adaptive-K={k}':<25} | {r['mean']:>8.3f} | {r['std']:>7.3f}  (gap vs A: {r['gap_vs_always_A']:+.3f}, p={r['p_vs_always_A']:.4f})")
    print("=" * 60)

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "adaptive_identification_results.json", "w") as f:
        json.dump({str(k) if isinstance(k, int) else k: v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to outputs/results/adaptive_identification_results.json")
