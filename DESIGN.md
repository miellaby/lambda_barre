# λ̄ Design — Sleep Cycle & Policy Training

Detailed implementation reference for the two-model brain: the offline sleep
pipeline (`Brain.sleep`) and policy training (`Brain.train_policy`). Numbers
and signatures live in `brain.py`; this document carries the order, the
reasoning, and the attention-mask geometry that are hard to reconstruct from
the code alone.

## Sleep Cycle Pipeline (`Brain.sleep`)

Dataset optimisation and World Model training first, then policy training.
The generator yields one phase step per UI frame, so the HUD/Dream Theater
follow the cycle live:

1. **Wake Extraction** — all sliding-window sequences of the wake session
   are extracted into the Addendum (`extract_addendum`).
2. **Active Forgetting Candidates** — a third of the Coreset is pruned into
   the Addendum (vivacity-biased random, `prune_coreset`): old memories are
   systematically put back in play at every sleep.
3. **Geometric Deduplication**
   - Intra-Addendum: complete-linkage clustering drops redundant duplicates
     (`deduplicate_addendum`).
   - Addendum vs Coreset: near-duplicates refresh the Coreset memory's
     vivacity and are dropped from the Addendum
     (`deduplicate_against_coreset`). Distances are saliency-weighted only
     once the coreset WM loss is low enough to trust the sensitivity map.
4. **World Model Training on Coreset** (`wm_coreset`) — establishes the
   baseline known dynamics. Runs until the epoch quota is validated AND the
   loss reaches the target. Each sequence's descent weight is proportional
   to its realized impact — the absolute change in innate cost between the
   decision state S4 and the terminal state S10 — so the WM focuses on
   sequences with real biological consequences; a small floor weight keeps
   neutral sequences learning at a trickle. The gradient is strongly
   boosted on the terminal salve's reward slots and on the EOS token.
   Causal passes alternate with EOS-lookahead passes (one lookahead in
   four): under the lookahead attention mask, the decision position (token
   76, whose head predicts a4) additionally attends the realized terminal
   EOS token (position 172), and the EOS token attends only the decision
   context and itself — so the WM learns the hindsight action (a4 given the
   context and the EOS) on the same trunk, without letting the recorded a4
   leak through the EOS key.
5. **Surprise Filtering** — inference only, no gradient: the WM predicts
   each surviving Addendum sequence. Error below threshold = familiar =
   dropped; error at or above = surprising = kept. UI stats update
   immediately, before Addendum training begins.
6. **World Model Training on Addendum** (`wm_addendum`) — same impact
   weighting and the same causal/lookahead alternation, on the surprising
   survivors.
7. **Policy Training** — supervised improvement by imitation of the WM's
   hindsight action, verified by imagined rollouts (`train_policy`); every
   step is recorded for the Dream Theater.
8. **Final Consolidation** — surviving Addendum sequences commit into the
   Coreset (`consolidate_addendum_into_coreset`), vivacity decay is applied,
   coldest memories are evicted above capacity.

## The EOS Token (dénouement)

Position 172 of a training sequence carries the denouement delta the
lookahead machinery conditions on. It is built at batch time (never recorded
in salves) from the innate cost channels — effort, douleur, courbature,
instabilité, vertige, confort — as the delta between the consequence salves
s5..s10 (the six states produced by a4..a9, i.e. exactly the salves the
imagination rolls out) and the decision salves s0..s4. The slots read
0.5 = neutral (no change), below = recovery, above = worsening; the scale is
calibrated on the measured dynamics so typical deltas use most of the range
and crisis spikes saturate.

- The slots stay raw per-channel deltas — the WM conditions on each
  channel's outcome. Wherever the EOS is read as ONE scalar (relief score,
  saliency anchor, Dream Theater display), the usual innate cost hierarchy
  applies: douleur above vertige above courbature/instabilité above
  effort/confort.
- Time is weighted inversely to the remaining time-to-denouement
  (hyperbolic), so the long-term outcome dominates and transient
  post-action costs count little.
- The interoception levels (fatigue, souffrance) are slow carriers at this
  horizon — quasi-static over a decision window — and are masked from every
  WM input.
- Structurally it is a dedicated token type (intero modality, canal
  orthogonal to the metabolic balance channel), so the WM can never confuse
  the outcome aggregate with a recorded interoception snapshot.

## Policy Training (`Brain.train_policy`)

Supervised policy improvement conditioned on World Model latent
representations. The default path imitates the WM's **hindsight action**
(EOS-lookahead supervision); the legacy stochastic-candidate machinery
survives as an explicit fallback.

- **Architecture & Input** (both paths): a 3×256 MLP; input = 45
  non-reward sensory scalars (proprioception, touch, vision, ball)
  concatenated with the normalized 64-dim intermediate latent of the World
  Model (109 floats total). Output: 5 motor consignes squashed into
  physical actuator ranges (tanh for angles, sigmoid for limb lengths).

- **Real Context Sampling** (both paths):
  - Batches are drawn from the buffer by prioritized sampling
    (`sample_for_policy`). Priority = the relief score: the flat mean of
    the cost channels over the decision window s0..s4 minus the
    time-weighted aggregate over s5..s10 (the same time weights as the
    EOS), scalarized under the usual innate cost hierarchy — i.e. the
    realized denouement delta; recovery trajectories dominate.
  - Context: 4 real past transitions plus the decision state s4 (77
    tokens). The WM inference extracts and normalizes the latent at s4 —
    the same latent the policy consumes at live wake, so training never
    feeds out-of-distribution latents.

- **Hindsight Supervision — EOS-lookahead** (default; about 3
  rollout-equivalents per batch instead of the legacy 6):
  - **Baseline rollout**: the policy's action acts on s4, then the policy
    reacts in closed loop to each imagined state for 6 steps (sliding
    77-token window). The accumulated cost is discounted (gamma > 1
    amplifies long-horizon consequences) and the imagined salves yield the
    predicted denouement delta (time-weighted consequence window minus the
    decision-window baseline read from the real context).
  - **Multiplicative forcing**: the predicted deviation from neutral is
    forced a few percent (random, per slot) further toward the favorable
    side — worsening shrinks toward neutral, recovery deepens — an automatic
    curriculum that scales with what the policy itself predicts.
  - **Hindsight generation**: a4 is generated in one forward on a compact
    [context, EOS] input — the EOS token carries the salve-10 positional
    encoding, so the decision query attends exactly the keys the training
    lookahead mask defines (mathematically identical conditioning).
  - **Verification forward**: causal rollout of the generated a4. One test,
    three roles: causality, reachability, graceful overshoot discovery.
  - **Comparative acceptance**: the generated a4 becomes the supervised
    target only if its verified cost beats the baseline cost by the
    exploration margin; otherwise no optimizer step. MSE toward accepted
    samples only, gradient norm clipped, Adam step.
  - **Fallbacks**: `--no-eos-lookahead` CLI flag; automatic fallback to the
    stochastic candidates when the horizon cannot reach the terminal salve
    or after consecutive fully-rejected batches.

- **Legacy Stochastic Candidates** (fallback, `_train_policy_candidates`):
  - 3 parallel candidates: policy baseline, recorded demonstration,
    Gaussian noise around the policy — all clamped to valid physical
    actuator ranges.
  - 2 future regimes over 6 imagined steps: Kalman/inertia (action
    velocity damped) and policy closed-loop.
  - Each imagined step incurs a biological cost; Bellman optimism takes
    the optimistic minimum across regimes.
  - A candidate must beat the baseline by the same exploration margin;
    the winner is detached as the supervised target.

- **Dream Theater Playback**:
  - Per-step history of the sleep cycle, browsable with a step selector
    (`< PREV` / `NEXT >` / `LIVE` buttons, arrow keys; `LIVE` keeps
    following the newest step).
  - Each hindsight step shows: the **lived sequence** s0→s10 (per-salve
    cost, realized EOS), the **WM completion** toward the forced EOS (real
    s0..s3 + generated a4 + imagined s5..s10), and the **decision**
    side-by-side — what the policy infers on s4 vs what the WM recommends,
    with costs, imagined EOS and accepted/rejected verdict.
  - Loss, acceptance rate and forced EOS in the header.
  - Legacy fallback records keep the candidate-by-regime storyboard,
    tab-filterable.
