The **$\bar{\lambda}$ (Lambda Barre)** project simulates a 2D biomechanical balancing creature powered by a dual neural network architecture interacting with a Pymunk physics environment:

- **World Model**: Transformer network predicting subsequent states, actions ($S_{t+1}, A_{t+1}). States include Biological costs.
- **Policy Network**: Actor network producing motor consignes ($\theta^*, d^*$) for front limbs, back limbs, and tail based on current sensor measures and World Model latent representation.
- **Sleep Cycle (`Brain.sleep`)**: Offline training consolidation combining active forgetting, surprise-based filtering, and imagined rollouts (Dream Theater).

## Memory Concepts & Terminology

| Concept | Definition | Rules & Invariants |
| :--- | :--- | :--- |
| **Salve** | A single snapshot of tokens $S_i + A_i$ produced at each World Model tick (3 Hz). | Composed of 13 state tokens + 3 action tokens (16 tokens $\times$ 25 floats). |
| **Sequence** | A temporal sliding window of consecutive salves (typically 10 salves). | All memory capacities are measured in **number of sequences**. |
| **Coreset** | Long-term consolidated memory | Actual ML Dataset of the World Model. Subject to vivacity decay and capacity eviction. |
| **Addendum** | Short-term memory | Receives recent wake experiences (sequences) and candidates revisited for active forgetting. |

Internal concepts (not for GUI)
* Buffer = (core, add, journal, ...)
* Journal = new experiences/sequences collection

**GUI Language** **must remain in English**.
**Token related are still in french** conversion in progress.

## Sleep Cycle Pipeline (`Brain.sleep`)

Dataset Optimisation & Word Model Training, then Policy Training:

1. **Wake Extraction**: Extract sliding-window sequences from the wake session into the Addendum (`extract_addendum`).
2. **Active Forgetting Candidates**: Randomly prune 33% of the Coreset (lowest vivacity biased random) into the Addendum (`prune_coreset`).
3. **Geometric Deduplication**:
   - **Intra-Addendum**: Complete linkage clustering removing redundant duplicates within the Addendum (`deduplicate_addendum`).
   - **Addendum vs. Coreset**: Cross-deduplication against the remaining Coreset (`deduplicate_against_coreset`). Sequences within distance $\epsilon < 0.04$ refresh the Coreset memory vivacity to 1.0 and are **immediately dropped from the Addendum**.
4. **World Model Training on Coreset**: Train the World Model on the remaining Coreset (`wm_coreset`) to establish baseline known dynamics (continues while $\text{loss} > 0.01$, stops when quota is validated and $\text{loss} \le 0.01$).
5. **Surprise Filtering (Cognitive Evaluation)**:
   - Evaluated in inference mode (`torch.no_grad()`), **without gradient descent**.
   - World Model predicts each surviving Addendum sequence.
   - Error $< \text{threshold}$ $\rightarrow$ familiar/known sequence $\rightarrow$ **dropped**.
   - Error $\ge \text{threshold}$ $\rightarrow$ surprising dynamic $\rightarrow$ **kept**.
   - **Update UI Stats immediately**: Update `filter_dropped` and compute expected `coreset_after` before Addendum training begins.
6. **World Model Training on Addendum**: Train the World Model on the surprising Addendum sequences (`wm_addendum`) with gradient descent (continues while $\text{loss} > 0.01$, stops when quota is validated and $\text{loss} \le 0.01$).
7. **Policy Training** Supervised policy improvement using imagined rollout trajectories in Dream Theater (`train_policy`).
8. **Final consolidation** Commit surviving Addendum sequences into the Coreset (`consolidate_addendum_into_coreset`), apply vivacity decay, and evict coldest memories if exceeding maximum capacity.

## Policy Training (`Brain.train_policy`)

Supervised policy improvement conditioned on World Model latent representations and guided by imagined rollouts in Dream Theater.

- **Architecture & Input**:
  - Multi-Layer Perceptron ($143 \to 256 \to 256 \to 256 \to 5$ with internal $\text{ReLU}$ activations).
  - Input: 47 normalized non-reward sensory scalars (proprioception, touch, vision, cursor) concatenated with the 96-dim intermediate latent vector of the World Model ($143$ floats total).
  - Output: 5 motor consignes $(\theta_{\text{front}}, d_{\text{front}}, \theta_{\text{back}}, d_{\text{back}}, \tau_{\text{tail}})$ squashed via $\tanh$ and $\text{sigmoid}$ into physical actuator ranges.


- **Real Context Sampling**:
  - Training draws batches from the experience buffer via prioritized sampling (`self.buffer.sample_for_policy(batch)`).
  - Prioritizes sequences demonstrating net biological recovery (EOS relief score $\text{relief} = -\Delta \text{souffrance} - 0.25 \cdot \Delta \text{fatigue}$) using stochastic weighted sampling without replacement ($w = \exp(2.0 \cdot \text{relief})$).
  - Context window: 4 real past transitions $[s_0, a_0, s_1, a_1, s_2, a_2, s_3, a_3, s_4]$ (77 tokens).
  - World Model inference extracts and normalizes the latent state representation at the decision state $s_4$.


- **Candidate Actions (4 parallel candidates)**:
  - Policy baseline: deterministic action from current policy network ($\pi$).
  - Recorded demonstration: ground-truth action from the sampled experience ($a_{\text{dataset}}$).
  - Low-noise exploration: Gaussian perturbation with $\sigma_1 = 0.15$.
  - High-noise exploration: Gaussian perturbation with $\sigma_2 = 0.30$.
  - All candidates are clamped to valid physical actuator limits.

- **Imagined Rollouts (3 future regimes over 6 steps / 2.0s at 3 Hz)**:
  - Each candidate action acts on $s_4$ to predict $s_5$ via `world.predict_next_state`.
  - From $s_5$, trajectories are rolled forward across 3 extrapolation regimes:
    - *Kalman / Inertia*: action continuation based on past action velocity vector $(a_2, a_3, a_4)$ damped by $0.8$ at each step.
    - *World Model Autoregressive*: World Model iteratively predicts both subsequent actions and states.
    - *Policy Closed-Loop*: Policy reacts dynamically to each imagined state via a sliding context window.

- **Bellman Optimism & Cost Evaluation**:
  - Each state step incurs a biological cost (`_salve_cost_batch`: mechanical power/effort, pain, fatigue, suffering, posture penalty).
  - Discounted cumulative cost: $C = c_0 + \sum_{k=1}^5 \gamma^k c_k$ with optimistic compounding $\gamma = 1.1$.
  - Bellman optimism takes the optimistic minimum across the 3 future regimes:
    $$C_{\text{cand}} = \min(C_{\text{Kalman}}, C_{\text{WM}}, C_{\text{Policy}})$$

- **Candidate Selection & Supervised Update**:
  - Stability margin: alternative candidates must achieve a $0.1\%$ cost reduction over the policy baseline to be selected, preventing policy jitter.
  - The winning candidate action is detached as a supervised target:
    $$\mathcal{L} = \text{MSE}(\pi(\text{pol\_in}), a^*_{\text{winner}})$$
  - Backpropagation with gradient norm clipping ($1.0$) and Adam optimizer step.

- **Dream Theater Playback**:
  - Live visualization in the sleep GUI of the 12 imagined trajectories (4 candidates $\times$ 3 regimes) for sample 0, highlighting the best regime per candidate and the winning action.


## UI standards

No **Font Antialiasing**, clean **Font Family**, **Window Resizing** with **Aspect Ratio**, **Fullscreen Friendly**, **Useful Metrics** with **Consistent Terminology**, **English Labels**

## File Map & Code Structure

`./lambda_barre/` folder:
* `brain.py`: Main brain engine (`Brain`), two-tier memory (`ExperienceBuffer`), sleep generator, surprise evaluation, vectorized policy rollouts.
* `main.py`: Pygame interactive loop, keyboard dispatcher, HUD display, live sleep stepper.
* `dream.py`: Visualizer for sleep policy rollouts (`DreamTheater`) and sleep training dialog.
* `wm_theater.py`: Standalone visualizer comparing ground-truth Coreset sequences with autoregressive World Model rollouts.
* `models.py`: PyTorch architectures for `WorldModel` and `Policy`.
* `tokenize.py`: 16-token dense salve representation (53 state signals + 5 motor actions).
* `body.py`, `world.py`, `proprio.py`, `extero.py`, `intero.py`: Physics simulation, joints, sensors, interoception, reward signals.
* `dataset.py`: Headless balance bootstrapping dataset generator.
* `test_brain.py`: Pytest test suite for end-to-end brain pipeline.

## Environment & Tooling

- **Python Virtual Environment** in `.venv` with ML/physics dependencies `torch`, `pygame`, `pymunk`
- Compile check: `.venv/bin/python -m py_compile <files>`.
- App run: `.venv/bin/python -m lambda_barre.main --accel`
- Torch **Checkpoints**: `buf_ckpt.pt` (experience buffer), `wm_ckpt.pt` (World Model), and `pol_ckpt.pt` (Policy)

## Rules for AI Agents

- **No Premature Tests**: The baseline is that the tests run well. **No need to test suites before code changes.**
- **No Unsolicited Numbers**: Unless ordering of information is crucial, **Avoid Numbered Lists**, **Chapter Numbers Are Bad**. Numbers are a mental burden with a maintenance cost. Avoid numbering as much as possible.
- **Source Filename is enough when documenting** (Avoid Pathes and file:// Links)

# Reads

- `README.md`
- Last session: `session_{date}.md` (chronological logs `session_YYYY_MM_DD.md` document daily decisions, invariant changes, and recent diagnoses)

