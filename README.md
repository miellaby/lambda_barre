# λ̄ (Lambda Barre)

**λ̄** is a **learning desktop-pet animat** project: an autonomous virtual animal with a simplified 2D biomechanical morphology, an autoregressive world model (Transformer), reinforcement learning over a realistic multimodal sensory field, and an innate reinforcement/comfort system.

---

## Installation

The project uses Python (>= 3.10) and `uv` (or `pip`):

```bash
# Create the virtual environment
uv venv
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e .
```

---

## Command line

### Launch

```bash
# Launch without any hardware acceleration, works on any CPU
.venv/bin/python -m lambda_barre.main

# Launch with maximum hardware acceleration (CUDA)
.venv/bin/python -m lambda_barre.main --accel

# Select the execution backend
.venv/bin/python -m lambda_barre.main --device cuda
.venv/bin/python -m lambda_barre.main --device auto
.venv/bin/python -m lambda_barre.main --device cpu
.venv/bin/python -m lambda_barre.main --device cpu-fast # MKL-DNN
```

The console script `lambda-barre` is equivalent to `.venv/bin/python -m lambda_barre.main`.

### Dream Theater (standalone)

Runs sleep cycles offline, straight from the checkpoints, without the physics simulation: the brain is restored, then a full sleep cycle unfolds in the Dream Theater (press `S` to start a cycle, `SPACE` to pause).

```bash
.venv/bin/python -m lambda_barre.dream_theater
dream-theater --accel               # console script equivalent
dream-theater --no-save             # view-only: checkpoints are not written back
```

| Option | Description | Possible values |
|---|---|---|
| `--wm-epochs N` | World Model training epochs per sleep cycle. | Integer *(default: 512)* |
| `--pol-steps N` | Policy training steps per sleep cycle. | Integer *(default: 32)* |
| `--no-save` | Skips writing the trained checkpoints back at the end of the cycle. | Boolean flag |
| `--no-eos-lookahead` | Falls back to the legacy stochastic candidates. | Boolean flag |
| `--wm` / `--pol` / `--buffer` | Checkpoint paths (defaults: repo `wm_ckpt.pt`, `pol_ckpt.pt`, `buf_ckpt.pt`). | Path |

### WM Theater

The World Model Theater (Coreset sequence inspection against WM rollouts) is available as `wm-theater`, equivalent to `.venv/bin/python -m lambda_barre.wm_theater`.

```bash
wm-theater                          # default checkpoints, relief sort
wm-theater --sort abs_desc          # extreme sequences first
```

### Options

| Option | Description | Possible values |
|---|---|---|
| `--accel` | Enables maximum hardware acceleration (alias for `--device auto`). | Boolean flag |
| `--device` | Specifies the compute hardware for the neural networks. | `cpu` *(default)*, `cuda`, `auto`, `cpu-fast` |
| `--reset` | Deletes existing checkpoints (`wm_ckpt.pt`, `pol_ckpt.pt`) and starts with fresh weights. | Boolean flag |
| `--headless` | Runs the simulation without a Pygame window (requires `--steps`). | Boolean flag |
| `--steps N` | Number of frames (60 Hz) to simulate in headless mode before stopping. | Integer (e.g. `300`) |

Options passed through the environment:

* **`LAMBDA_ACCEL=1`**: `--accel`.
* **`LAMBDA_DEVICE=cuda`**: `--device cuda`
* **`LAMBDA_TORCH_MKLDNN=1`**: `--device cpu-fast`
* **`SDL_VIDEODRIVER=dummy`**: Ideal for headless mode.

### Configuration file (`config.toml`)

Checkpoint-structure options are **not** exposed on the CLI. They define the
shape of `wm_ckpt.pt`, `pol_ckpt.pt`, and `buf_ckpt.pt` — changing any of them
invalidates existing checkpoints. Set them in `config.toml` instead.

| TOML key | Parameter | Affected checkpoint |
|---|---|---|
| `[transformer] d_model` | Transformer latent dimension | `wm_ckpt.pt`, `pol_ckpt.pt` |
| `[transformer] nhead` | Attention heads | `wm_ckpt.pt` |
| `[transformer] layers` | Transformer encoder layers | `wm_ckpt.pt` |
| `[transformer] dim_ff` | Feed-forward hidden dimension | `wm_ckpt.pt` |
| `[transformer] dropout` | Dropout rate | `wm_ckpt.pt` (training behaviour) |
| `[policy] hidden` | Policy hidden layer width | `pol_ckpt.pt` |
| `[sequence] seq_steps` | Transitions per training sequence | `buf_ckpt.pt`, `wm_ckpt.pt` |
| `[sequence] n_ctx` | Decision context length | `buf_ckpt.pt`, `wm_ckpt.pt` |
| `[sequence] wm_hz` | World Model recording cadence (Hz) | `buf_ckpt.pt` (temporal semantics) |
| `[sequence] pol_hz` | Policy cadence (Hz, typically 2× wm_hz) | `buf_ckpt.pt` (temporal semantics) |
| `[buffer] capacity` | Session journal capacity (sequences) | `buf_ckpt.pt` (truncation) |
| `[buffer] pool_capacity` | Coreset long-term capacity (sequences) | `buf_ckpt.pt` (truncation) |

After changing any of these, run with `--reset` to discard old checkpoints
and start fresh:

```bash
.venv/bin/python -m lambda_barre.main --reset
```

## Controls (GUI)

| Key / Action | Function |
|---|---|
| **`ESC`** | Quit |
| **`SPACE`** or **`B`** | *brain off/on* |
| **`S`** | Trigger a **Sleep** phase (offline training) |
| **`R`** | **Reset**: puts the animat back at its spawn position |
| **`G`** | Hide motor consignes |
| **Joysticks** | Drive the animat in manual mode |
| **Right-click + drag on platform** | Move a platform |
| **Mouse** | Move the cursor perceived by the animat |

---

## Architecture & Temporal cadences

```
               [ Physics Simulation 60 Hz ]
            Pymunk 2D, Contacts, Forces
                             │
            ┌────────────────┴────────────────┐
            ▼ (3 Hz)                          ▼ (6 Hz)
    [ World Model ]                      [ Policy ]
 Autoregressive Transformer             MLP Actor
 Predicts token salves            Motor consignes (5)
  (13 states + 3 actions)       (Front/back limbs, tail)
```

* **Physics & Sensors (60 Hz)**: Pymunk articulated skeleton and sensors: proprioception, exteroception and interoception.
* **World Model (3 Hz)**: Dense tokenization of 16 tokens per salve and 25 dimensions per token. A causal autoregressive model that imagines a 2-second horizon.
* **Policy (6 Hz)**: Selects the motor action from 45 sensory scalars and the latent state extracted from the World Model.

During the **Sleep phase (`S`)**: offline training of the World Model (self-supervised prediction of the future) and of the Policy (candidate tournament guided by the futures imagined by the World Model).

---

## Tests

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH=. .venv/bin/pytest
```
