"""
Train shared PPO policies on simple_spread_v3 (N=3, discrete actions).

Uses parameter sharing: one policy network shared across all agents.
Each agent's obs is treated as an independent sample in the same rollout.
Different seeds → different conventions.

Saves: outputs/policies/policy_A.pt, policy_B.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from coord_env import make_env as make_coord_env, N_AGENTS, OBS_DIM, ACT_DIM, MAX_CYCLES
from pathlib import Path


# ---------------------------------------------------------------------------
# Gym wrapper: treats each agent step as an independent env sample
# ---------------------------------------------------------------------------
class SimpleSpreadSingleAgentWrapper(gym.Env):
    """
    Wraps coord_env parallel env as a single-agent gym env.
    Each reset/step cycles through agents round-robin.
    The shared policy sees one agent's obs and outputs one agent's action.
    All agents share the same policy (parameter sharing).
    """

    def __init__(self, n_agents=N_AGENTS, max_cycles=MAX_CYCLES, seed=None):
        super().__init__()
        self.n_agents   = n_agents
        self.max_cycles = max_cycles
        self._seed      = seed

        # Spaces (discrete actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACT_DIM)

        self._make_env()

    def _make_env(self):
        self.env = make_coord_env(max_cycles=self.max_cycles)
        self._pending_obs  = {}
        self._agent_queue  = []
        self._step_actions = {}
        self._ep_reward    = 0.0
        self._done         = False

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed or self._seed)
        self._pending_obs  = dict(obs)
        self._agent_queue  = list(self.env.agents)
        self._step_actions = {}
        self._ep_reward    = 0.0
        self._done         = False
        self._current_agent = self._agent_queue[0]
        return self._pending_obs[self._current_agent].astype(np.float32), {}

    def step(self, action):
        self._step_actions[self._current_agent] = int(action)
        agent_idx = self._agent_queue.index(self._current_agent)

        if agent_idx < len(self._agent_queue) - 1:
            # More agents to collect actions for — return current obs, zero reward
            self._current_agent = self._agent_queue[agent_idx + 1]
            next_obs = self._pending_obs[self._current_agent].astype(np.float32)
            return next_obs, 0.0, False, False, {}
        else:
            # All agents have actions — step the env
            obs, rewards, terms, truncs, _ = self.env.step(self._step_actions)
            self._pending_obs  = dict(obs)
            self._agent_queue  = list(self.env.agents) if self.env.agents else []
            self._step_actions = {}

            done     = not self.env.agents or any(terms.values()) or any(truncs.values())
            reward   = float(np.mean(list(rewards.values()))) if rewards else 0.0
            self._ep_reward += reward

            if done or not self._agent_queue:
                self._done = True
                # Return dummy obs on done
                return np.zeros(OBS_DIM, dtype=np.float32), reward, done, done, {}
            else:
                self._current_agent = self._agent_queue[0]
                next_obs = self._pending_obs[self._current_agent].astype(np.float32)
                return next_obs, reward, False, False, {}

    def close(self):
        self.env.close()


class ReturnLogCallback(BaseCallback):
    def __init__(self, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.ep_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.log_freq == 0 and self.ep_rewards:
            mean_r = np.mean(self.ep_rewards[-50:])
            print(f"  step {self.num_timesteps:>7d} | mean_ep_return={mean_r:.2f}")
        return True


def make_env(seed=None):
    def _init():
        env = SimpleSpreadSingleAgentWrapper(seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init


def train_policy(pop_name: str, seed: int, total_timesteps: int = 500_000):
    print(f"\n{'='*55}")
    print(f"Training {pop_name} (seed={seed}, steps={total_timesteps})")
    print(f"{'='*55}")

    n_envs = 8
    vec_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
    )

    cb = ReturnLogCallback(log_freq=50_000)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    out_dir = Path("outputs/policies")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / pop_name))
    print(f"  Saved to outputs/policies/{pop_name}.zip")

    vec_env.close()
    return model


def eval_policy(pop_name: str, n_episodes: int = 100):
    """Evaluate saved policy and measure convention (landmark assignment)."""
    from stable_baselines3 import PPO as PPO_load

    model = PPO_load.load(f"outputs/policies/{pop_name}")

    env = make_coord_env()
    assignment = np.zeros((N_AGENTS, N_AGENTS))  # assignment[agent_i][landmark_j]
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        tot = 0.0
        for _ in range(MAX_CYCLES):
            if not env.agents: break
            actions = {}
            for agent in env.agents:
                act, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = int(act)
            obs, rews, _, _, _ = env.step(actions)
            tot += sum(rews.values()) / len(rews)
        returns.append(tot)

        # Record final positions
        try:
            raw = env.aec_env.env.env
            world = raw.world
            agent_pos = np.array([a.state.p_pos for a in world.agents])
            lm_pos    = np.array([l.state.p_pos for l in world.landmarks])
            for j, lp in enumerate(lm_pos):
                nearest = np.argmin([np.linalg.norm(agent_pos[i] - lp) for i in range(N_AGENTS)])
                assignment[nearest][j] += 1
        except Exception:
            pass

    env.close()
    assignment /= max(n_episodes, 1)

    print(f"\n{pop_name} — eval return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Convention matrix P(agent_i covers landmark_j):")
    print("         LM0    LM1")
    for i in range(N_AGENTS):
        print(f"  agent_{i}: {assignment[i,0]:.2f}   {assignment[i,1]:.2f}")
    return assignment, np.mean(returns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 7, 99])
    args = parser.parse_args()

    # Train all seeds, then pick the most divergent pair
    convs = {}
    for seed in args.seeds:
        name = f"policy_seed{seed}"
        train_policy(name, seed=seed, total_timesteps=args.steps)
        conv, ret = eval_policy(name)
        convs[name] = (conv, ret)

    # Find most divergent pair
    names = list(convs.keys())
    best_div, best_pair = 0, (names[0], names[1])
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = np.abs(convs[names[i]][0] - convs[names[j]][0]).mean()
            print(f"  {names[i]} vs {names[j]}: divergence={d:.3f}")
            if d > best_div:
                best_div, best_pair = d, (names[i], names[j])

    print(f"\nBest pair: {best_pair[0]} vs {best_pair[1]} (divergence={best_div:.3f})")

    # Save best pair as policy_A and policy_B
    import shutil
    shutil.copy(f"outputs/policies/{best_pair[0]}.zip", "outputs/policies/policy_A.zip")
    shutil.copy(f"outputs/policies/{best_pair[1]}.zip", "outputs/policies/policy_B.zip")
    print("Saved best pair as policy_A and policy_B")
