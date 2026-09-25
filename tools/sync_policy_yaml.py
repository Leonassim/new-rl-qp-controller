#!/usr/bin/env python3
"""Regenerate the policy-dependent YAML blocks from a trained ONNX + mjlab.

Emits `ref_joint_order`, `q0` and `action_scale` ready to paste into
etc/NewRLQPController.in.yaml, and cross-checks every value it can against a
second source. It does NOT edit the YAML: that file carries a lot of hand-written
rationale in comments, and splicing blocks into it automatically would lose them.

Run it with mjlab's interpreter, which is what makes the cross-check possible:

    cd <mjlab> && .venv/bin/python <this repo>/tools/sync_policy_yaml.py \\
        --pdgains <hrp5p_mj_description>/pdgains/PDgains_sim.dat \\
        /path/to/new_policy.onnx

Why not read everything from the ONNX metadata:
  - `action_scale` is stored rounded to 3 decimals, so copying it would quantise
    a value the deployment needs at full precision. mjlab's HRP5P_ACTION_SCALE is
    the authoritative source; the metadata is used to VERIFY it.
  - `joint_stiffness` / `joint_damping` are placeholders in the exporter (37
    entries of 1.000 / -0.000 against 53 joints), so kp/kd are checked against
    PDgains_sim.dat instead, never taken from the ONNX.

Exit code is non-zero if any cross-check fails, so this can gate a deployment.
"""

import argparse
import math
import sys
from pathlib import Path

import onnx

# Joints the controller drives but the policy does not: refJointOrder keeps them
# (dof()==1 and hasJoint()), so configRL() demands they be present in q0, but the
# network never writes them. Held at the robot module's half-sitting value.
UNDRIVEN_Q0 = {"RHDY": 0.0, "LHDY": 0.0}


def meta(model: onnx.ModelProto) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument(
        "--pdgains",
        type=Path,
        help="hrp5p_mj_description/pdgains/PDgains_sim.dat, the file mc_mujoco loads. "
        "Omitted, kp/kd go unverified and the run says so.",
    )
    args = ap.parse_args()

    try:
        from mjlab.asset_zoo.robots.hrp5.hrp5_constants import (
            HRP5P_ACTION_SCALE,
            HRP5P_INIT_STATE,
            HRP5P_PD_GAINS,
        )
        from mjlab.envs.mdp.observations import ACTUATED_HRP5
    except ImportError:
        print("Run this with mjlab's interpreter, see the module docstring.", file=sys.stderr)
        return 2

    model = onnx.load(str(args.onnx))
    md = meta(model)
    names = md["joint_names"].split(",")
    default_pos = [float(v) for v in md["default_joint_pos"].split(",")]
    scale_meta = [float(v) for v in md["action_scale"].split(",")]

    # The action set, taken from the policy's own indices rather than assumed.
    order = [names[i] for i in ACTUATED_HRP5]
    onnx_default = dict(zip(names, default_pos))

    problems: list[str] = []

    if len(order) != len(scale_meta):
        problems.append(
            f"action_scale has {len(scale_meta)} entries but ACTUATED_HRP5 selects "
            f"{len(order)} joints -- the exporter and mjlab disagree on the action set"
        )

    obs = md.get("observation_names", "")
    n_obs_terms = len(obs.split(",")) if obs else 0
    hist = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    # 3 vector terms of width 3, then joint_pos/joint_vel/actions of width len(order).
    per_frame = 3 * 4 + 3 * len(order)
    if hist % per_frame:
        problems.append(f"obs width {hist} is not a multiple of {per_frame} (one frame)")
    else:
        print(f"# obs {hist} = {hist // per_frame} frames x {per_frame}  ({n_obs_terms} terms: {obs})")

    # action_scale: full precision from mjlab, verified against the rounded metadata.
    for n, s_meta in zip(order, scale_meta):
        s = HRP5P_ACTION_SCALE[n]
        if abs(round(s, 3) - s_meta) > 1e-9:
            problems.append(f"action_scale[{n}]: mjlab {s:.6f} -> {round(s, 3)} != onnx {s_meta}")

    # q0 likewise: mjlab's init state is the full-precision source.
    init = dict(HRP5P_INIT_STATE.joint_pos)
    for n in order:
        q_mj = init.get(n, 0.0)
        if abs(round(q_mj, 3) - onnx_default[n]) > 1e-9:
            problems.append(f"q0[{n}]: mjlab {q_mj:.6f} -> {round(q_mj, 3)} != onnx {onnx_default[n]}")

    # kp/kd against the file mc_mujoco actually loads, never against the ONNX.
    if args.pdgains is None:
        print("# kp/kd NOT verified: pass --pdgains to check them against PDgains_sim.dat")
    elif not args.pdgains.exists():
        problems.append(f"{args.pdgains} not found")
    else:
        rows = [ln.split() for ln in args.pdgains.read_text().splitlines() if ln.strip()]
        dat = dict(zip(names, [(float(a), float(b)) for a, b in rows]))
        for n, (kp, kd) in HRP5P_PD_GAINS.items():
            if n in dat and (abs(kp - dat[n][0]) > 1e-6 or abs(kd - dat[n][1]) > 1e-6):
                problems.append(f"kp/kd[{n}]: mjlab {(kp, kd)} != PDgains_sim.dat {dat[n]}")

    if problems:
        print("\nCROSS-CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    run = md.get("run_path", "?")
    print(f"# generated from {args.onnx.name} (run {run}) -- all cross-checks passed\n")

    print("    ref_joint_order:")
    for n in order:
        print(f"      - {n}")

    print("\n    q0:")
    width = max(len(n) for n in list(order) + list(UNDRIVEN_Q0))
    for n in order:
        q = init.get(n, 0.0)
        deg = f"    # {math.degrees(q):.2f} deg" if abs(q) > 1e-9 else ""
        print(f"      {n + ':':{width + 1}} {q:.6f}{deg}")
    for n, q in UNDRIVEN_Q0.items():
        print(f"      {n + ':':{width + 1}} {q:.6f}")

    print("\n    action_scale:")
    for n in order:
        print(f"      {n + ':':{width + 1}} {HRP5P_ACTION_SCALE[n]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
