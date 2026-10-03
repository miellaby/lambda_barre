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
