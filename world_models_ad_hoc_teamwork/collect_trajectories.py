"""
Collect trajectories for world model training.

Records (obs_self, obs_partner, action_self, next_obs_self) tuples.
The key difference from v1: we record next_obs_self, not partner action.
The world model predicts next_obs_self from (obs_self, obs_partner, action_self).

Output:
  outputs/trajectories/traj_A.npz
  outputs/trajectories/traj_B.npz
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from coord_env import make_env, N_AGENTS, MAX_CYCLES


def collect_trajectories(policy_path: str, n_episodes: int, seed_offset: int = 0):
    model = PPO.load(policy_path)
    env   = make_env()

    all_obs_self      = []
    all_obs_partner   = []
    all_actions       = []
    all_next_obs_self = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)

        for _ in range(MAX_CYCLES):
            if not env.agents:
                break
            agents  = list(env.agents)
            actions = {}
            for agent in agents:
                act, _ = model.predict(obs[agent], deterministic=False)
                actions[agent] = int(act)

            next_obs, _, _, _, _ = env.step(actions)

            # Record each agent as ego, other as partner
            for i, agent in enumerate(agents):
                for j, partner in enumerate(agents):
                    if i == j:
                        continue
                    if agent in next_obs:
                        all_obs_self.append(obs[agent].copy())
                        all_obs_partner.append(obs[partner].copy())
                        all_actions.append(actions[agent])
                        all_next_obs_self.append(next_obs[agent].copy())

            obs = next_obs

        if (ep + 1) % 500 == 0:
            print(f"  Episode {ep+1}/{n_episodes}, transitions: {len(all_actions)}")

    env.close()

    return (
        np.array(all_obs_self,      dtype=np.float32),
        np.array(all_obs_partner,   dtype=np.float32),
        np.array(all_actions,       dtype=np.int64),
        np.array(all_next_obs_self, dtype=np.float32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path("outputs/trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pop in ["A", "B"]:
        print(f"\nCollecting trajectories for Population {pop}...")
        obs_self, obs_partner, actions, next_obs_self = collect_trajectories(
            policy_path=f"outputs/policies/policy_{pop}",
            n_episodes=args.n_episodes,
            seed_offset=0 if pop == "A" else 100000,
        )
        path = out_dir / f"traj_{pop}.npz"
        np.savez_compressed(
            str(path),
            obs_self=obs_self,
            obs_partner=obs_partner,
            actions=actions,
            next_obs_self=next_obs_self,
        )
        print(f"  Total transitions: {len(actions)}")
        print(f"  Saved to {path}")
