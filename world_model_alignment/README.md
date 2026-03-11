# World Model Alignment

This folder contains the experiments studying whether coordination-relevant predictive consistency depends on **alignment between learned world models**. The central question is whether two agents need only individually useful predictive models, or whether they also benefit from having internal models that agree with each other about how the environment evolves.

The experiments use Dreamer-style recurrent state-space world models trained on the MineRL ObtainDiamond dataset. Each model is trained to predict future observations, rewards, and continuation signals from sequences of states and actions. To control alignment, the experiments introduce a training-overlap parameter `alpha`. When `alpha = 1`, the two models are trained on identical trajectory data. When `alpha = 0`, they are trained on disjoint trajectory data. Intermediate values correspond to partial overlap. This creates a clean experimental manipulation of how much shared experience the two models receive.

Alignment is measured through **rollout disagreement**. Starting from the same real initial state and applying the same action sequence, both models simulate the future for a fixed rollout horizon `H`. Disagreement is defined as the mean squared error between the two predicted states after `H` steps. In this setup, higher disagreement means the models imagine different futures even under identical conditions.

The main finding is that increased overlap between training data substantially reduces predictive disagreement between world models. This effect is consistent across rollout horizons. When there is no shared training data, disagreement is 0.280 at horizon 1, 0.401 at horizon 5, 0.410 at horizon 10, and 0.420 at horizon 20. When the models are trained on identical data, disagreement drops to 0.142, 0.239, 0.252, and 0.265 at the same horizons. These correspond to reductions of 49%, 40%, 38%, and 37%, respectively. The pattern also shows that disagreement grows with horizon, reflecting the way small predictive differences compound over multi-step simulation. :contentReference[oaicite:4]{index=4}

The experiments also separate disagreement on milestone transitions from disagreement on ordinary background transitions. Milestones are timesteps with positive reward and correspond to important task events such as collecting resources or crafting progress. The results show that most divergence arises from modeling general environment dynamics rather than major reward events. At horizon 10, milestone disagreement falls from 0.094 to 0.071 when moving from `alpha = 0` to `alpha = 1`, while background disagreement falls much more sharply from 0.410 to 0.253. This indicates that world-model misalignment is primarily expressed in the dense, ordinary parts of the trajectory rather than in sparse high-level events. :contentReference[oaicite:5]{index=5}

A coordination-proxy analysis is also included. In this analysis, each model uses its internal predictions to evaluate candidate actions, and the experiment measures how often both models choose the same action. This signal is much noisier than predictive disagreement and does not produce a clean monotonic alignment effect. The interpretation is that planning decisions amplify small predictive differences, making action-level agreement unstable. For this reason, predictive disagreement is treated as the stronger and more reliable measure of alignment in this project.

Finally, the folder includes an alignment-repair experiment. A misaligned model is fine-tuned on small amounts of shared calibration data to test whether predictive compatibility can be restored. The results show partial recovery: disagreement falls from 0.410 with no shared calibration data to 0.392 with 2,000 shared sequences and to 0.383 with 5,000 shared sequences, with performance plateauing around 10,000 sequences. This suggests that alignment is not fixed after independent training and can be partially repaired through shared experience. :contentReference[oaicite:6]{index=6}

Overall, the experiments in this folder support the conclusion that world-model alignment strongly affects predictive consistency. Shared training experience produces more compatible internal dynamics models, disagreement grows over longer horizons when alignment is weak, and small amounts of shared data can partially repair misalignment. The broader implication is that coordination between AI agents may depend not only on the quality of each agent’s model in isolation, but also on whether different agents maintain compatible predictive representations of the environment.

The notebooks in this folder are organized as a pipeline: dataset exploration, sequence construction, baseline and Dreamer-style model training, overlap-conditioned alignment sweeps, rollout disagreement evaluation, coordination-proxy analysis, repair experiments, and final plot generation. Together they document the full experimental design and reproduce the results reported in the project writeup.

## Main notebooks

- `01_dataset_exploration.ipynb`
- `02_sequence_dataset_builder.ipynb`
- `03_train_baseline_world_model.ipynb`
- `04_train_rssm_dreamer_world_model.ipynb`
- `05_alignment_experiment_sweep.ipynb`
- `06_rollout_evaluation.ipynb`
- `07_coordination_proxy_analysis.ipynb`
- `08_alignment_repair_experiment.ipynb`
- `09_generate_plots_and_tables.ipynb`
