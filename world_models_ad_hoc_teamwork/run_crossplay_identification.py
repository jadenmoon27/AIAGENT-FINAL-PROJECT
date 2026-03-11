"""
Cross-play identification: can WM prediction error identify partner population
when the ego agent is always from Pop-A (mixed-policy setting)?

Tests two error signals:
  1. Full obs error (indices 0:10) — what we used before, fails at ~50%
  2. other_rel only (indices 8:10) — partner-movement-only signal,
     filters out ego-distribution noise

Also tests the cross-population WM approach:
  - Collect (Pop-A ego, Pop-A partner) data → train WM-AA
  - Collect (Pop-A ego, Pop-B partner) data → train WM-AB
  - Both in-distribution for ego, differ only in partner convention

Runs identification in the mixed setting:
  agent 0 = Pop-A policy (fixed)
  agent 1 = Pop-A policy OR Pop-B policy (unknown)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from train_world_models import load_world_model, train_world_model, save_world_model, WorldModel
from coord_env import make_env, MAX_CYCLES, OBS_DIM, ACT_DIM

PARTNER_REL_SLICE = slice(8, 10)  # other_rel indices in 10-dim obs


def run_identification_mixed(policy_ego, policy_A, policy_B, wm_A, wm_B,
                              n_episodes=500, k_values=None,
                              error_slice=None):
    """
    agent 0: always policy_ego (Pop-A)
    agent 1: half Pop-A, half Pop-B
    error_slice: if set, only compare prediction error on those obs indices
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 8, 10, 15, 20]

    results = {k: {"correct": 0, "total": 0} for k in k_values}

    for ep in range(n_episodes):
        use_pop_A_partner = (ep % 2 == 0)
        partner_policy    = policy_A if use_pop_A_partner else policy_B
        true_pop          = "A" if use_pop_A_partner else "B"

        env = make_env()
        obs, _ = env.reset(seed=ep)
        agents = list(env.agents)
        if len(agents) < 2:
            env.close()
            continue

        agent0, agent1 = agents[0], agents[1]
        err_A, err_B = 0.0, 0.0
        step = 0

        for _ in range(MAX_CYCLES):
            if not env.agents:
                break

            actions = {}
            # agent 0: always ego policy
            if agent0 in env.agents:
                act, _ = policy_ego.predict(obs[agent0], deterministic=False)
                actions[agent0] = int(act)
            # agent 1: partner policy
            if agent1 in env.agents:
                act, _ = partner_policy.predict(obs[agent1], deterministic=False)
                actions[agent1] = int(act)

            prev_obs = {a: obs[a].copy() for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
            step += 1

            if agent0 not in obs or agent1 not in prev_obs:
                break

            pred_A = wm_A.predict_np(prev_obs[agent0], prev_obs[agent1], actions.get(agent0, 0))
            pred_B = wm_B.predict_np(prev_obs[agent0], prev_obs[agent1], actions.get(agent0, 0))
            actual = obs[agent0]

            if error_slice is not None:
                err_A += np.linalg.norm(pred_A[error_slice] - actual[error_slice])
                err_B += np.linalg.norm(pred_B[error_slice] - actual[error_slice])
            else:
                err_A += np.linalg.norm(pred_A - actual)
                err_B += np.linalg.norm(pred_B - actual)

            if step in k_values:
                predicted = "A" if err_A <= err_B else "B"
                results[step]["correct"] += int(predicted == true_pop)
                results[step]["total"]   += 1

        env.close()

    return results


def collect_crossplay_trajectories(policy_ego, policy_partner, n_episodes, seed_offset=0):
    """Collect (ego-obs, partner-obs, ego-action, next-ego-obs) from mixed-policy episodes."""
    obs_self_list, obs_partner_list, actions_list, next_obs_list = [], [], [], []

    env = make_env()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        agents = list(env.agents)
        if len(agents) < 2:
            continue
        agent0, agent1 = agents[0], agents[1]

        for _ in range(MAX_CYCLES):
            if not env.agents:
                break
            actions = {}
            if agent0 in env.agents:
                act, _ = policy_ego.predict(obs[agent0], deterministic=False)
                actions[agent0] = int(act)
            if agent1 in env.agents:
                act, _ = policy_partner.predict(obs[agent1], deterministic=False)
                actions[agent1] = int(act)

            prev = {a: obs[a].copy() for a in env.agents}
            obs, _, _, _, _ = env.step(actions)

            if agent0 in obs and agent1 in prev:
                obs_self_list.append(prev[agent0])
                obs_partner_list.append(prev[agent1])
                actions_list.append(actions.get(agent0, 0))
                next_obs_list.append(obs[agent0])

    env.close()
    return (np.array(obs_self_list, dtype=np.float32),
            np.array(obs_partner_list, dtype=np.float32),
            np.array(actions_list, dtype=np.int64),
            np.array(next_obs_list, dtype=np.float32))


def print_results(label, results, k_values):
    print(f"\n  {label}")
    print(f"  {'K':>4} | {'Accuracy':>9} | {'N':>5}")
    print(f"  {'-'*25}")
    for k in k_values:
        r = results[k]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"  {k:>4} | {acc:>9.3f} | {r['total']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",    type=int, default=500)
    parser.add_argument("--traj-episodes", type=int, default=3000)
    parser.add_argument("--wm-epochs",     type=int, default=100)
    parser.add_argument("--device",        type=str, default="cpu")
    parser.add_argument("--skip-retrain",  action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("CROSS-PLAY IDENTIFICATION EXPERIMENT")
    print("="*60)

    policy_A = PPO.load("outputs/policies/policy_A")
    policy_B = PPO.load("outputs/policies/policy_B")
    wm_A = load_world_model("outputs/world_models/wm_A.pt", device=args.device)
    wm_B = load_world_model("outputs/world_models/wm_B.pt", device=args.device)

    k_values = [1, 2, 3, 5, 8, 10, 15, 20]

    # ------------------------------------------------------------------
    # Test 1: Full obs error in mixed setting (baseline — expect ~50%)
    # ------------------------------------------------------------------
    print("\n--- Test 1: Full obs error (mixed setting, expect ~50%) ---")
    res_full = run_identification_mixed(
        policy_A, policy_A, policy_B, wm_A, wm_B,
        n_episodes=args.n_episodes, k_values=k_values, error_slice=None
    )
    print_results("Full obs error", res_full, k_values)

    # ------------------------------------------------------------------
    # Test 2: other_rel only (indices 8:10)
    # ------------------------------------------------------------------
    print("\n--- Test 2: other_rel only (indices 8:10) ---")
    res_rel = run_identification_mixed(
        policy_A, policy_A, policy_B, wm_A, wm_B,
        n_episodes=args.n_episodes, k_values=k_values,
        error_slice=PARTNER_REL_SLICE
    )
    print_results("other_rel error only", res_rel, k_values)

    # ------------------------------------------------------------------
    # Test 3: Cross-population WMs (retrain with same ego, different partner)
    # ------------------------------------------------------------------
    out_dir = Path("outputs/world_models")
    wm_AA_path = out_dir / "wm_AA.pt"
    wm_AB_path = out_dir / "wm_AB.pt"
    wm_BB_path = out_dir / "wm_BB.pt"
    wm_BA_path = out_dir / "wm_BA.pt"

    if not args.skip_retrain:
        print(f"\n--- Collecting cross-play trajectories ({args.traj_episodes} eps each) ---")

        print("  (Pop-A ego + Pop-A partner)...")
        os_AA, op_AA, ac_AA, no_AA = collect_crossplay_trajectories(
            policy_A, policy_A, args.traj_episodes, seed_offset=0)
        print(f"  Collected {len(ac_AA)} transitions")

        print("  (Pop-A ego + Pop-B partner)...")
        os_AB, op_AB, ac_AB, no_AB = collect_crossplay_trajectories(
            policy_A, policy_B, args.traj_episodes, seed_offset=100000)
        print(f"  Collected {len(ac_AB)} transitions")

        print("  (Pop-B ego + Pop-B partner)...")
        os_BB, op_BB, ac_BB, no_BB = collect_crossplay_trajectories(
            policy_B, policy_B, args.traj_episodes, seed_offset=200000)
        print(f"  Collected {len(ac_BB)} transitions")

        print("  (Pop-B ego + Pop-A partner)...")
        os_BA, op_BA, ac_BA, no_BA = collect_crossplay_trajectories(
            policy_B, policy_A, args.traj_episodes, seed_offset=300000)
        print(f"  Collected {len(ac_BA)} transitions")

        print("\n  Training WM-AA (Pop-A ego, Pop-A partner)...")
        wm_AA, info_AA = train_world_model(os_AA, op_AA, ac_AA, no_AA,
                                            epochs=args.wm_epochs, verbose=True)
        save_world_model(wm_AA, wm_AA_path)
        print(f"  WM-AA val loss: {info_AA['best_val_loss']:.6f}")

        print("\n  Training WM-AB (Pop-A ego, Pop-B partner)...")
        wm_AB, info_AB = train_world_model(os_AB, op_AB, ac_AB, no_AB,
                                            epochs=args.wm_epochs, verbose=True)
        save_world_model(wm_AB, wm_AB_path)
        print(f"  WM-AB val loss: {info_AB['best_val_loss']:.6f}")

        print("\n  Training WM-BB (Pop-B ego, Pop-B partner)...")
        wm_BB, info_BB = train_world_model(os_BB, op_BB, ac_BB, no_BB,
                                            epochs=args.wm_epochs, verbose=True)
        save_world_model(wm_BB, wm_BB_path)
        print(f"  WM-BB val loss: {info_BB['best_val_loss']:.6f}")

        print("\n  Training WM-BA (Pop-B ego, Pop-A partner)...")
        wm_BA, info_BA = train_world_model(os_BA, op_BA, ac_BA, no_BA,
                                            epochs=args.wm_epochs, verbose=True)
        save_world_model(wm_BA, wm_BA_path)
        print(f"  WM-BA val loss: {info_BA['best_val_loss']:.6f}")

        # Save val loss metadata for quality-confound check
        import json as _json
        with open(out_dir / "wm_crosspop_metadata.json", "w") as f:
            _json.dump({
                "WM_AA": info_AA['best_val_loss'], "WM_AB": info_AB['best_val_loss'],
                "WM_BB": info_BB['best_val_loss'], "WM_BA": info_BA['best_val_loss'],
            }, f, indent=2)
        print("\n  Val losses (quality check — should be comparable within ego group):")
        print(f"    WM-AA={info_AA['best_val_loss']:.6f}  WM-AB={info_AB['best_val_loss']:.6f}  (Pop-A ego)")
        print(f"    WM-BB={info_BB['best_val_loss']:.6f}  WM-BA={info_BA['best_val_loss']:.6f}  (Pop-B ego)")
    else:
        print("\n--- Loading existing cross-pop WMs ---")
        wm_AA = load_world_model(wm_AA_path, device=args.device)
        wm_AB = load_world_model(wm_AB_path, device=args.device)
        wm_BB = load_world_model(wm_BB_path, device=args.device)
        wm_BA = load_world_model(wm_BA_path, device=args.device)

    print("\n--- Test 3: Cross-pop WMs (WM-AA vs WM-AB, same ego dist) ---")
    res_cross = run_identification_mixed(
        policy_A, policy_A, policy_B, wm_AA, wm_AB,
        n_episodes=args.n_episodes, k_values=k_values, error_slice=None
    )
    print_results("Cross-pop WMs (full obs)", res_cross, k_values)

    res_cross_rel = run_identification_mixed(
        policy_A, policy_A, policy_B, wm_AA, wm_AB,
        n_episodes=args.n_episodes, k_values=k_values,
        error_slice=PARTNER_REL_SLICE
    )
    print_results("Cross-pop WMs (other_rel only)", res_cross_rel, k_values)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY at K=10")
    for label, res in [("Full obs (original WMs)", res_full),
                        ("other_rel only (original WMs)", res_rel),
                        ("Cross-pop WMs (full obs)", res_cross),
                        ("Cross-pop WMs (other_rel)", res_cross_rel)]:
        r = res[10]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"  {label:<40}: {acc:.3f}")
    print(f"  {'Random baseline':<40}: 0.500")
    print("="*60)

    out_dir2 = Path("outputs/results")
    out_dir2.mkdir(parents=True, exist_ok=True)
    with open(out_dir2 / "crossplay_identification_results.json", "w") as f:
        json.dump({
            "full_obs": {str(k): v for k, v in res_full.items()},
            "other_rel": {str(k): v for k, v in res_rel.items()},
            "cross_pop_full": {str(k): v for k, v in res_cross.items()},
            "cross_pop_rel": {str(k): v for k, v in res_cross_rel.items()},
        }, f, indent=2)
    print(f"\nSaved to outputs/results/crossplay_identification_results.json")
