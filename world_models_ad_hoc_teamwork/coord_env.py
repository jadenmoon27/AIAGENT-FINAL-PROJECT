"""
coord_env.py — Coupled coordination environment.

Key changes from v1:
  1. Simultaneous coverage reward: reward only fires when BOTH agents are
     simultaneously near their respective landmarks under the best assignment.
     This makes the ego agent's reward fundamentally dependent on partner trajectory.
  2. Momentum coupling: agents within DRAG_RADIUS exert drag on each other,
     making transitions physically coupled (not just reward-coupled).

These two changes make accurate partner prediction load-bearing:
  - Without knowing where/when the partner will arrive, the planner cannot
    optimize for simultaneous coverage.
  - A world model trained on one population's data implicitly encodes that
    population's timing and assignment conventions.

Observation (per agent, 10-dim): same as v1
  vel(2) + pos(2) + lm_rel(2x2=4) + other_rel(1x2=2) = 10
"""

import numpy as np
from pettingzoo.mpe.simple_spread.simple_spread import raw_env as _BaseRawEnv
from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import aec_to_parallel

N_AGENTS     = 2
N_LANDMARKS  = 2
OBS_DIM      = 10   # vel(2)+pos(2)+lm_rel(4)+other_rel(2)
ACT_DIM      = 5
MAX_CYCLES   = 25

# Reward tuning
COVER_RADIUS  = 0.3   # distance threshold for "covering" a landmark
SIMUL_BONUS   = 2.0   # bonus when BOTH agents simultaneously cover their landmarks
SOLO_PENALTY  = 0.5   # penalty per landmark NOT covered each step

# Momentum coupling
DRAG_RADIUS   = 0.4   # distance within which agents drag each other
DRAG_COEFF    = 0.15  # fraction of relative velocity transferred


class CoupledCoordScenario(BaseScenario):
    def make_world(self, N=2):
        world = World()
        world.dim_c = 2
        world.collaborative = True
        world.agents = [Agent() for _ in range(N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        world.landmarks = [Landmark() for _ in range(N)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = False
            lm.movable = False
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for lm in world.landmarks:
            lm.color = np.array([0.25, 0.25, 0.25])

        # Place agents randomly
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Landmarks equidistant from mean agent position
        mean_pos = np.mean([a.state.p_pos for a in world.agents], axis=0)
        offset = np_random.uniform(0.3, 0.7)
        angle = np_random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        world.landmarks[0].state.p_pos = mean_pos + offset * direction
        world.landmarks[1].state.p_pos = mean_pos - offset * direction
        for lm in world.landmarks:
            lm.state.p_pos = np.clip(lm.state.p_pos, -1, 1)
            lm.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, a1, a2):
        return np.linalg.norm(a1.state.p_pos - a2.state.p_pos) < (a1.size + a2.size)

    def reward(self, agent, world):
        """Per-agent collision penalty."""
        rew = 0.0
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1.0
        return rew

    def global_reward(self, world):
        """
        Simultaneous coverage reward.
        Find best landmark assignment, give full bonus only when both covered.
        """
        positions = [a.state.p_pos for a in world.agents]
        lm_pos    = [lm.state.p_pos for lm in world.landmarks]

        # Try both assignments: (a0→lm0, a1→lm1) and (a0→lm1, a1→lm0)
        d00 = np.linalg.norm(positions[0] - lm_pos[0])
        d11 = np.linalg.norm(positions[1] - lm_pos[1])
        d01 = np.linalg.norm(positions[0] - lm_pos[1])
        d10 = np.linalg.norm(positions[1] - lm_pos[0])

        # Best assignment by total distance
        if d00 + d11 <= d01 + d10:
            dist_a, dist_b = d00, d11
        else:
            dist_a, dist_b = d01, d10

        covered_a = dist_a < COVER_RADIUS
        covered_b = dist_b < COVER_RADIUS

        rew = 0.0
        if covered_a and covered_b:
            # Both simultaneously covered: big bonus minus residual distances
            rew += SIMUL_BONUS - dist_a - dist_b
        else:
            # Partial: penalize uncovered landmarks
            rew -= dist_a + dist_b
            if not covered_a:
                rew -= SOLO_PENALTY
            if not covered_b:
                rew -= SOLO_PENALTY

        return rew

    def observation(self, agent, world):
        lm_pos = [lm.state.p_pos - agent.state.p_pos
                  for lm in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos
                     for other in world.agents if other is not agent]
        return np.concatenate(
            [agent.state.p_vel, agent.state.p_pos] + lm_pos + other_pos
        )


def make_env(max_cycles=MAX_CYCLES, render_mode=None):
    """Return a parallel_env with the coupled coordination scenario."""
    scenario = CoupledCoordScenario()
    world = scenario.make_world(N=N_AGENTS)

    env = _BaseRawEnv(
        N=N_AGENTS,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.scenario = scenario
    env.world = world
    env.reset()

    par_env = aec_to_parallel(env)
    return par_env


# ---------------------------------------------------------------------------
# Analytical physics with momentum coupling
# ---------------------------------------------------------------------------
SENSITIVITY = 5.0
DAMPING     = 0.25
DT          = 0.1


def physics_step(vel, pos, action_int):
    """Standard MPE physics (no coupling — use physics_step_coupled for that)."""
    ux = (1 if action_int == 2 else 0) - (1 if action_int == 1 else 0)
    uy = (1 if action_int == 4 else 0) - (1 if action_int == 3 else 0)
    force   = SENSITIVITY * np.array([ux, uy], dtype=np.float32)
    new_vel = vel * (1.0 - DAMPING) + force * DT
    new_pos = pos + new_vel * DT
    return new_vel, new_pos


def physics_step_coupled(vel_self, pos_self, action_self,
                          vel_other, pos_other):
    """
    Physics step with momentum coupling.
    When agents are within DRAG_RADIUS, they exchange a fraction of
    relative velocity — simulating physical accommodation.
    """
    ux = (1 if action_self == 2 else 0) - (1 if action_self == 1 else 0)
    uy = (1 if action_self == 4 else 0) - (1 if action_self == 3 else 0)
    force   = SENSITIVITY * np.array([ux, uy], dtype=np.float32)
    new_vel = vel_self * (1.0 - DAMPING) + force * DT

    # Drag coupling
    dist = np.linalg.norm(pos_self - pos_other)
    if dist < DRAG_RADIUS and dist > 1e-6:
        drag = DRAG_COEFF * (vel_other - vel_self) * (1.0 - dist / DRAG_RADIUS)
        new_vel = new_vel + drag

    new_pos = pos_self + new_vel * DT
    return new_vel, new_pos


def parse_obs(obs):
    """Parse 10-dim obs into (vel, pos, lm_abs_list, other_pos_list)."""
    vel = obs[0:2].copy()
    pos = obs[2:4].copy()
    lm_abs    = [pos + obs[4 + 2*i : 6 + 2*i] for i in range(N_LANDMARKS)]
    other_pos = [pos + obs[8 + 2*i : 10 + 2*i] for i in range(N_AGENTS - 1)]
    return vel, pos, lm_abs, other_pos


def build_obs(vel, pos, lm_abs, other_poss):
    """Reconstruct 10-dim obs."""
    return np.concatenate([
        vel, pos,
        *[lm - pos for lm in lm_abs],
        *[op - pos for op in other_poss],
    ]).astype(np.float32)


def simultaneous_coverage_reward(pos_self, pos_other, lm_positions):
    """Analytical reward for WM rollouts."""
    positions = [pos_self, pos_other]
    lm_pos    = lm_positions

    d00 = np.linalg.norm(positions[0] - lm_pos[0])
    d11 = np.linalg.norm(positions[1] - lm_pos[1])
    d01 = np.linalg.norm(positions[0] - lm_pos[1])
    d10 = np.linalg.norm(positions[1] - lm_pos[0])

    if d00 + d11 <= d01 + d10:
        dist_a, dist_b = d00, d11
    else:
        dist_a, dist_b = d01, d10

    covered_a = dist_a < COVER_RADIUS
    covered_b = dist_b < COVER_RADIUS

    if covered_a and covered_b:
        return SIMUL_BONUS - dist_a - dist_b
    else:
        r = -(dist_a + dist_b)
        if not covered_a: r -= SOLO_PENALTY
        if not covered_b: r -= SOLO_PENALTY
        return r
