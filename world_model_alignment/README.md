# World Model Alignment

This folder contains experiments studying whether **alignment between learned world models** affects predictive consistency between agents.

## Core Question

If two agents use learned dynamics models, is it enough that each model is useful on its own?

Or do the models also need to **agree with each other** about how the environment evolves?

---

## Setup

We train **two Dreamer-style world models** on the **MineRL ObtainDiamond** dataset.

Each model predicts:
- next observation
- reward
- continuation / termination

To control alignment, we vary how much training data the two models share.

### Overlap parameter

- `alpha = 0` → completely different training trajectories
- `alpha = 1` → identical training trajectories
- `alpha = 0.25, 0.5, 0.75` → partial overlap

Higher `alpha` means the two models are trained on more similar experience.

---

## Metric

We measure **predictive disagreement**.

Procedure:
1. start from the same real state
2. apply the same action sequence
3. let both models simulate the future
4. compare their predicted states

Disagreement is the MSE between the two predicted states after horizon `H`.

### Rollout horizons

- `H = 1`
- `H = 5`
- `H = 10`
- `H = 20`

Longer horizons are harder because prediction errors compound over time.

---

## Main Results

Shared training experience strongly reduces disagreement.

| Horizon | alpha = 0 | alpha = 1 | Reduction |
|--------|-----------|-----------|-----------|
| 1      | 0.280     | 0.142     | 49%       |
| 5      | 0.401     | 0.239     | 40%       |
| 10     | 0.410     | 0.252     | 38%       |
| 20     | 0.420     | 0.265     | 37%       |

### Interpretation

- More overlap → more aligned world models
- More aligned models → less predictive divergence
- Disagreement grows with horizon, but alignment reduces it consistently

---

## Milestone vs Background Analysis

We also separate disagreement into:

- **milestone transitions**: important reward-bearing events
- **background transitions**: ordinary environment dynamics

At horizon 10:

- milestone disagreement: `0.094 → 0.071`
- background disagreement: `0.410 → 0.253`

Most disagreement comes from ordinary background dynamics rather than major task events.

---

## Coordination Proxy

We also test a simple planning-based coordination proxy:
- each model evaluates candidate actions
- we measure how often both models choose the same action

This signal is noisy and less reliable than predictive disagreement, so the main conclusion of this folder is based on rollout disagreement.

---

## Repair Experiment

We test whether misalignment can be repaired by fine-tuning one model on small amounts of shared data.

| Shared sequences | Disagreement |
|------------------|--------------|
| 0                | 0.410        |
| 2000             | 0.392        |
| 5000             | 0.383        |
| 10000            | 0.385        |

This shows that alignment can be **partially recovered** through shared experience.

---

## Main Takeaways

- Training overlap strongly affects world-model alignment
- Alignment strongly affects predictive agreement
- Misalignment becomes worse over longer rollout horizons
- Shared calibration data can partially repair misalignment

---

## Notebook Order

Run notebooks in this order:

1. `01_dataset_exploration.ipynb`
2. `02_sequence_dataset_builder.ipynb`
3. `03_train_baseline_world_model.ipynb`
4. `04_train_rssm_dreamer_world_model.ipynb`
5. `05_alignment_experiment_sweep.ipynb`
6. `06_rollout_evaluation.ipynb`
7. `07_coordination_proxy_analysis.ipynb`
8. `08_alignment_repair_experiment.ipynb`
9. `09_generate_plots_and_tables.ipynb`
