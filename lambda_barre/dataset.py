"""Dataset generation and bootstrapping for the λ̄ creature.

Generates rich physical simulation experience across diverse regimes:
  - passive balance & natural micro-dynamics
  - gentle external swaying forces
  - coordinated motor variations of limbs and tail
  - recovery from dynamic perturbations

Usage as a script:
    python -m lambda_barre.dataset --transitions 3000 --out buf_ckpt.pt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
import torch

from . import body as B, world as W, proprio as S, extero as E, intero as I
from .brain import Brain, ExperienceBuffer
from .tokenize import DenseEncoder

WM_DT = 1.0 / 3.0  # 3 Hz world model recording cadence


def generate_balance_dataset(
    target_transitions: int = 3000,
    seed: int = 42,
    progress_callback: callable | None = None,
) -> ExperienceBuffer:
    """Run headless physical simulation to collect ``target_transitions`` of
    diverse balancing experience.

    Returns an ``ExperienceBuffer`` pre-populated with sliding-window
    training sequences in its persistent replay pool.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    space = W.make_space()
    skel = B.build_skeleton(space)
    B.apply_consignes(skel)

    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = DenseEncoder()

    buffer = ExperienceBuffer(capacity=1000, pool_capacity=1000, seed=seed)

    collected = 0
    falls = 0

    regimes = ["passive", "sway", "motor_variation", "recovery"]
    regime_idx = 0
    regime_timer = 0.0
    regime_duration = 20.0  # switch physical regime every 20s of simulation

    frame = 0
    dt = 1.0 / 60.0
    accum = 0.0

    while collected < target_transitions:
        regime = regimes[regime_idx % len(regimes)]
        regime_timer += dt
        if regime_timer >= regime_duration:
            regime_timer = 0.0
            regime_idx += 1
            buffer.boundary()

        # Regime-specific dynamics
        if regime == "passive":
            # Neutral standing
            skel.limb_l.theta_star = skel.spawn_theta_l
            skel.limb_r.theta_star = skel.spawn_theta_r
            skel.limb_l.d_star = skel.spawn_d_l
            skel.limb_r.d_star = skel.spawn_d_r
            skel.tail_act.theta_star = 0.0

        elif regime == "sway":
            # Gentle sinusoidal force on torso inducing realistic sway
            omega = 1.5 + 0.5 * math.sin(frame * 0.01)
            force_x = 35.0 * math.sin(omega * (frame * dt))
            skel.torso.apply_force_at_local_point((force_x, 0), (0, 0))

        elif regime == "motor_variation":
            # Coordinated limb adjustments
            t_s = frame * dt
            d_mod = 3.0 * math.sin(2.0 * t_s)
            skel.limb_l.d_star = max(B.LIMB_MIN, min(B.LIMB_MAX, skel.spawn_d_l + d_mod))
            skel.limb_r.d_star = max(B.LIMB_MIN, min(B.LIMB_MAX, skel.spawn_d_r - d_mod))
            th_mod = 0.12 * math.sin(1.2 * t_s)
            skel.limb_l.theta_star = skel.spawn_theta_l + th_mod
            skel.limb_r.theta_star = skel.spawn_theta_r + th_mod
            skel.tail_act.theta_star = -0.2 * math.sin(1.2 * t_s)

        elif regime == "recovery":
            # Occasional small impulsive nudge, testing stabilization
            if frame % 180 == 0:
                impulse = random.choice([-25.0, 25.0])
                skel.torso.apply_impulse_at_local_point((impulse, 0), (0, 20))

        # Physics simulation (substeps)
        touch.reset_contacts()
        for _ in range(3):
            W.step(space, skel, 1.0 / 180.0)
        frame += 1

        # Fall check: reset cleanly without corrupting the sequence
        angle_deg = abs(skel.torso.angle) * 180.0 / math.pi
        if angle_deg > 60.0 or skel.torso.position.y < 25.0:
            falls += 1
            buffer.boundary()
            B.reset(skel)
            proprio.reset()
            reward.reset()
            W.reset_ball(space)
            cursor.reset()
            vision.reset()
            touch.reset()
            intero.reset()
            B.apply_consignes(skel)
            continue

        sig = proprio.update(skel, dt)
        ts = touch.update(skel, dt)
        rs = reward.update(skel, sig, ts, dt)
        isg = intero.update(rs, dt)
        cs = cursor.update(skel, getattr(space, "ball", (400, 300)), dt)
        vs = vision.update(skel, dt)

        accum += dt
        if accum >= WM_DT:
            accum -= WM_DT
            salve = encoder.encode(sig, ts, cs, vs, isg, rs, skel)
            buffer.push(salve)
            collected += 1
            if progress_callback is not None:
                progress_callback(collected, target_transitions)

    # Consolidate linear journal into persistent pool with dense stride 1
    buffer.boundary()
    buffer.consolidate(stride=1)
    return buffer


def main() -> None:
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "buf_ckpt.pt"
    )
    parser = argparse.ArgumentParser(description="Generate 100x balance dataset for lambda_barre")
    parser.add_argument("--transitions", type=int, default=3000,
                        help="number of transitions to record (default: 3000 = ~2500 sequences, 100x bigger)")
    parser.add_argument("--out", type=str, default=default_out,
                        help=f"output file path (default: {default_out})")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Generating {args.transitions} transitions of physical experience...", flush=True)

    def on_progress(cur, total):
        if cur % 500 == 0 or cur == total:
            print(f"  [{cur}/{total}] ({cur/total*100:.0f}%) transitions collected...", flush=True)

    buf = generate_balance_dataset(target_transitions=args.transitions, seed=args.seed,
                                  progress_callback=on_progress)
    buf.save(args.out)
    elapsed = time.perf_counter() - t0
    file_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Dataset generated in {elapsed:.2f}s:", flush=True)
    print(f"  Replay pool sequences: {buf.pool_size}", flush=True)
    print(f"  Saved to: {args.out} ({file_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
