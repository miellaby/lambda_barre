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

## Sleep Cycle & Policy Training

Offline training (`Brain.sleep`): dataset optimization first, then World Model training, then Policy training. The Policy improves by imitating the World Model's hindsight (EOS-lookahead) action; a stochastic-candidate machinery survives as its fallback (`--no-eos-lookahead`).

Full mechanics — active forgetting, geometric deduplication, impact-proportional World Model training with EOS-lookahead passes, surprise filtering, hindsight supervision, candidate fallback, Dream Theater playback — are documented in `DESIGN.md`.

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
- **Temporary scripts** (scratch checks, throwaway experiments) go in the `.tmp/` folder (gitignored), never at the repo root.
- Torch **Checkpoints**: `buf_ckpt.pt` (experience buffer), `wm_ckpt.pt` (World Model), and `pol_ckpt.pt` (Policy)

## Rules for AI Agents

- **No Premature Tests**: The baseline is that the tests run well. **No need to test suites before code changes.**
- **No Unsolicited Numbers**: Unless ordering of information is crucial, **Avoid Numbered Lists**, **Chapter Numbers Are Bad**. Numbers are a mental burden with a maintenance cost. Avoid numbering as much as possible.
- **Source Filename is enough when documenting** (Avoid Pathes and file:// Links)

# Reads

- `README.md`
- `DESIGN.md` (implementation details: sleep cycle pipeline, policy training)
- Last session: `session_{date}.md` (chronological logs `session_YYYY_MM_DD.md` document daily decisions, invariant changes, and recent diagnoses)
