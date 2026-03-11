# AI Agents Final Project

Final project for **COSC 89.34 – AI Agents** at Dartmouth College.

## Team
- Alexander Almanza
- Jaden Moon
- Arthur Ufongene

## Overview

This repository contains two complementary experiments about **world models in multi-agent systems**.

The first studies **world-model alignment**: whether two agents need compatible internal dynamics models in order to predict the future consistently. The second studies **world models as convention detectors**: whether a world model trained on multi-agent trajectory data can identify a partner’s behavioral convention and support ad hoc teamwork.

Together, the project asks a broader question:

> How do learned world models affect coordination between agents?

---

## Repository Structure

```text
AIAGENT_FINAL_PROJECT
│
├── world_model_alignment/
│   Dreamer-style world model experiments on MineRL
│
└── alex_multiwalker_experiment/
    Convention detection and ad hoc teamwork experiments
```
## Part 1: World Model Alignment

The `world_model_alignment/` folder contains experiments using Dreamer-style world models trained on the **MineRL ObtainDiamond** dataset.

Two world models are trained while varying how much training data they share. This is controlled by an overlap parameter:

- `alpha = 0` → no shared training data
- `alpha = 1` → identical training data
- intermediate values → partial overlap

The two models are then rolled forward from the same initial state under the same action sequence, and their predicted future states are compared. The main metric is **predictive disagreement**.

### Main result

More shared training experience leads to much lower disagreement between world models.

| Horizon | alpha = 0 | alpha = 1 | Reduction |
|--------|-----------|-----------|-----------|
| 1      | 0.280     | 0.142     | 49%       |
| 5      | 0.401     | 0.239     | 40%       |
| 10     | 0.410     | 0.252     | 38%       |
| 20     | 0.420     | 0.265     | 37%       |

This shows that shared experience makes learned dynamics models substantially more compatible. A repair experiment also shows that misalignment can be partially reduced by fine-tuning one model on a small amount of shared calibration data.

---

## Part 2: Convention Detection in Ad Hoc Teamwork

The `alex_multiwalker_experiment/` folder contains experiments studying whether world models implicitly encode partner conventions in a cooperative multi-agent setting.

The main idea is that independently trained populations often develop different coordination conventions. A world model trained on trajectory data may capture those differences, allowing an agent to identify which kind of partner it is interacting with.

The main contribution in this part is an **ego-matched training protocol** that isolates partner-convention information from ego-policy noise.

### Main result

- Naive partner identification: approximately **50%** (chance)
- Ego-matched identification after 10 steps: **91.6%**

This identification is useful for coordination: once the partner’s convention is identified, the agent can switch to the matching policy and achieve performance statistically indistinguishable from oracle behavior.

A further result is that simply planning directly with aligned world models does **not** reliably improve coordination. The main benefit comes from convention detection and policy adaptation, not shared imagination alone.

---

## Why These Two Experiments Belong Together

These two folders study different aspects of the same big idea.

- `world_model_alignment/` shows that shared experience controls predictive compatibility
- `alex_multiwalker_experiment/` shows that world models can encode socially meaningful structure such as partner conventions

Taken together, the repository argues that world models matter in multi-agent systems not just because they predict the environment, but because they shape how agents relate to one another in cooperative settings.

---

## Notes

This repository is organized for the final project submission. Large datasets and heavyweight model artifacts may not be included directly.

For more details on each experiment, see the README inside each folder.
