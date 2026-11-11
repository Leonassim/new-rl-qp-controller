#!/usr/bin/env python3
"""cylinder_limits.py: real (roll, pitch) workspace of RHPS1's cylinder-driven hips and ankles.

Kinematics = AngleToCylinder.cpp (RSSR, closed form) with the attachment points of RHPS1.in.conf
(parallel_cylinder_points), BC/BD of rhp_joint_settings.h and the joint chain of RHPS1main.urdf
(roll then pitch about intersecting axes). A pose is feasible when both cylinder lengths stay inside the
stroke (URDF prismatic limits) minus a margin. Cuts each corner of the URDF box with a tangent half-plane (conservative) and writes them as mc_rtc
CompoundJointConstraintDescription ((q - p1) x (p2 - p1) <= 0, q = [roll, pitch]).

  python3 scripts/cylinder_limits.py [--margin-mm 3] [--out limits]
"""
import argparse
import math

import numpy as np

BC, BD = 0.07, 0.06  # rod (LT) and crank (LC), RHPS1 block of rhp_joint_settings.h (crotch = ankle)
JOINTS = {
  # name: (pos0 on base0, pos2 on base1, pos3 on base0, stroke [m], force [N], URDF box roll, URDF box pitch)
  "hip": dict(pos0=(0.0, 0.035, -0.29), pos2=(-0.05, 0.035, -0.055), pos3=(0.0, 0.035, -0.05),
              stroke=(0.19478, 0.29133), force=1550.0, roll=(-0.628319, 0.628319), pitch=(-1.91986, 0.610865)),
  "ankle": dict(pos0=(0.024, 0.029, -0.075), pos2=(-0.04924, 0.029, -0.008682), pos3=(0.0, 0.029, -0.317),
                stroke=(0.19891, 0.29279), force=776.0, roll=(-0.523599, 0.523599), pitch=(-1.5708, 0.698132)),
}


def rx(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def base1_point_in_base0(kind, p2, r, p):
  """pos2 (on base1) expressed in base0. Hip: base0 = thigh (CROTCH_P), base1 = CROTCH_Y, RP centre 0.055 below
  CROTCH_Y. Ankle: base0 = shank (KNEE_P), base1 = foot (ANKLE_P), RP centre 0.36 below the knee."""
  R = rx(r) @ ry(p)
  if kind == "hip":
    return R.T @ (np.asarray(p2) - np.array([0.0, 0.0, -0.055]))
  return np.array([0.0, 0.0, -0.36]) + R @ np.asarray(p2)


def cyl_length(kind, side, r, p):
  """AngleToCylinder.cpp, parallel cylinder branch. side +1 / -1 picks the y sign of the attachment points."""
  d = JOINTS[kind]
  flip = np.array([1.0, side, 1.0])
  pos0, pos2, pos3 = (np.asarray(d[k]) * flip for k in ("pos0", "pos2", "pos3"))
  Xc, Yc, Zc = base1_point_in_base0(kind, pos2, r, p) - pos3
  s = Xc * Xc + Yc * Yc + Zc * Zc - BC * BC + BD * BD
  with np.errstate(invalid="ignore"):
    if abs(Xc) > abs(Zc):
      alp, bet = s / (2 * Xc), -Zc / Xc
      gam, dl = bet * alp, alp * alp - BD * BD
      sq = math.sqrt(gam * gam - (bet * bet + 1) * dl) if gam * gam - (bet * bet + 1) * dl >= 0 else float("nan")
      z0, z1 = (-gam + sq) / (bet * bet + 1), (-gam - sq) / (bet * bet + 1)
      x0, x1 = alp + bet * z0, alp + bet * z1
    else:
      alp, bet = s / (2 * Zc), -Xc / Zc
      gam, dl = bet * alp, alp * alp - BD * BD
      sq = math.sqrt(gam * gam - (bet * bet + 1) * dl) if gam * gam - (bet * bet + 1) * dl >= 0 else float("nan")
      x0, x1 = (-gam + sq) / (bet * bet + 1), (-gam - sq) / (bet * bet + 1)
      z0, z1 = alp + bet * x0, alp + bet * x1
  if x0 < 0 and not math.isnan(x0):
    q = pos3 + np.array([x0, 0.0, z0])
  elif not math.isnan(x1):
    q = pos3 + np.array([x1, 0.0, z1])
  else:
    return float("nan")
  return float(np.linalg.norm(pos0 - q))


def lengths(kind, r, p):
  return cyl_length(kind, +1, r, p), cyl_length(kind, -1, r, p)


def jacobian(kind, r, p, h=1e-5):
  """dL/d(roll, pitch), 2x2 (rows: cylinder +y, -y)."""
  J = np.zeros((2, 2))
  for j, (dr, dp) in enumerate(((h, 0.0), (0.0, h))):
    a = np.array(lengths(kind, r + dr, p + dp))
    b = np.array(lengths(kind, r - dr, p - dp))
    J[:, j] = (a - b) / (2 * h)
  return J


def feasible_map(kind, margin, rr, pp):
  lo, hi = JOINTS[kind]["stroke"]
  ok = np.zeros((len(pp), len(rr)), bool)
  L = np.full((len(pp), len(rr), 2), np.nan)
  for i, p in enumerate(pp):
    for j, r in enumerate(rr):
      l = lengths(kind, r, p)
      L[i, j] = l
      ok[i, j] = all(not math.isnan(x) and lo + margin <= x <= hi - margin for x in l)
  return ok, L

def constraints(kind, margin):
  """The four stroke constraints as h(q) >= 0: (cylinder side, sign, bound)."""
  lo, hi = JOINTS[kind]["stroke"]
  return [(s, +1, lo + margin) for s in (+1, -1)] + [(s, -1, hi - margin) for s in (+1, -1)]


def h_val(kind, c, r, p):
  s, sg, b = c
  l = cyl_length(kind, s, r, p)
  return float("nan") if math.isnan(l) else sg * (l - b)


def box_samples(kind, n=90):
  d = JOINTS[kind]
  rr = np.linspace(d["roll"][0], d["roll"][1], n)
  pp = np.linspace(d["pitch"][0], d["pitch"][1], n)
  return np.stack(np.meshgrid(rr, pp), -1).reshape(-1, 2)


def tangent_cut(kind, c, samples):
  """Best conservative half-plane n.(q - q0) >= 0 for constraint c, among the tangents at boundary points and the
  chords between two boundary points (a convex boundary admits no conservative tangent): kept only if no sample
  of the box on its side violates c; the one keeping the most samples wins. None when c never binds in the box."""
  d = JOINTS[kind]
  hv = np.array([h_val(kind, c, r, p) for r, p in samples])
  bad = ~(hv >= 0)
  if not bad.any():
    return None
  good_ref = samples[np.argmax(np.where(np.isfinite(hv), hv, -np.inf))]
  pts, grads = [], []
  for r0 in np.linspace(d["roll"][0] - 0.2, d["roll"][1] + 0.2, 57):
    pp = np.linspace(d["pitch"][0] - 0.3, d["pitch"][1] + 0.3, 400)
    hs = np.array([h_val(kind, c, r0, p) for p in pp])
    idx = np.nonzero(np.isfinite(hs[:-1]) & np.isfinite(hs[1:]) & (np.sign(hs[:-1]) != np.sign(hs[1:])))[0]
    for i in idx:
      lo_, hi_ = pp[i], pp[i + 1]
      for _ in range(40):
        mid = 0.5 * (lo_ + hi_)
        if np.sign(h_val(kind, c, r0, mid)) == np.sign(hs[i]):
          lo_ = mid
        else:
          hi_ = mid
      q0 = np.array([r0, 0.5 * (lo_ + hi_)])
      e = 1e-5
      g = np.array([(h_val(kind, c, r0 + e, q0[1]) - h_val(kind, c, r0 - e, q0[1])) / (2 * e),
                    (h_val(kind, c, r0, q0[1] + e) - h_val(kind, c, r0, q0[1] - e)) / (2 * e)])
      pts.append(q0); grads.append(g)
  cands = [(q0, g) for q0, g in zip(pts, grads)]
  for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
      t = pts[j] - pts[i]
      n = np.array([-t[1], t[0]])
      if (good_ref - pts[i]) @ n < 0:
        n = -n
      cands.append((pts[i], n))
  best = None
  for q0, n in cands:
    nn = np.linalg.norm(n)
    if nn == 0:
      continue
    n = n / nn
    keep = (samples - q0) @ n >= 0
    if (keep & bad).any():
      continue
    if best is None or keep.sum() > best[2]:
      best = (q0, n, keep.sum())
  if best is None:
    raise RuntimeError(f"{kind}: no conservative cut for constraint {c}")
  return best


def clip(poly, q0, n):
  """Sutherland-Hodgman: keep n.(q - q0) >= 0."""
  out = []
  for k in range(len(poly)):
    a, b = poly[k], poly[(k + 1) % len(poly)]
    fa, fb = (a - q0) @ n, (b - q0) @ n
    if fa >= 0:
      out.append(a)
    if fa * fb < 0:
      out.append(a + (b - a) * fa / (fa - fb))
  return np.array(out)


def polygon(kind, margin):
  d = JOINTS[kind]
  samples = box_samples(kind)
  poly = np.array([[d["roll"][0], d["pitch"][0]], [d["roll"][1], d["pitch"][0]],
                   [d["roll"][1], d["pitch"][1]], [d["roll"][0], d["pitch"][1]]])  # CCW
  cuts = []
  for c in constraints(kind, margin):
    t = tangent_cut(kind, c, samples)
    if t is None:
      continue
    q0, n, _ = t
    cuts.append((c, q0, n))
    poly = clip(poly, q0, n)
  return poly, cuts


def check(kind, margin, poly, n=150):
  """Fraction of a dense sample of the polygon that violates the stroke (should be 0)."""
  from matplotlib.path import Path
  path = Path(poly)
  rr = np.linspace(poly[:, 0].min(), poly[:, 0].max(), n)
  pp = np.linspace(poly[:, 1].min(), poly[:, 1].max(), n)
  G = np.stack(np.meshgrid(rr, pp), -1).reshape(-1, 2)
  G = G[path.contains_points(G, radius=-1e-9)]
  lo, hi = JOINTS[kind]["stroke"]
  v = sum(not all(not math.isnan(x) and lo + margin - 1e-6 <= x <= hi - margin + 1e-6 for x in lengths(kind, r, p))
          for r, p in G)
  return v, len(G)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--margin-mm", type=float, default=3.0)
  ap.add_argument("--out", default="limits")
  a = ap.parse_args()
  m = a.margin_mm / 1000.0
  for kind, ref in (("hip", 0.2274937), ("ankle", 0.2384281)):
    l = lengths(kind, 0.0, 0.0)
    print(f"{kind}: neutral lengths {l[0]*1000:.4f} / {l[1]*1000:.4f} mm (binary {ref*1000:.4f})")
  res = {}
  names = {"hip": ("CROTCH_R", "CROTCH_P"), "ankle": ("ANKLE_R", "ANKLE_P")}
  yaml = ["# mc_rtc CompoundJointConstraintDescription: (q - p1) x (p2 - p1) <= 0, q = [j1, j2] = [roll, pitch] [rad]",
          f"# RHPS1 cylinder strokes minus {a.margin_mm:g} mm (cyl_limits.py); the URDF box stays in the joint limits.",
          "compoundJoints:"]
  for kind in ("hip", "ankle"):
    d = JOINTS[kind]
    poly, cuts = polygon(kind, m)
    v, nG = check(kind, m, poly)
    rr = np.linspace(-1.2, 1.2, 241)
    pp = np.linspace(d["pitch"][0] - 0.5, d["pitch"][1] + 0.5, 301)
    ok, L = feasible_map(kind, m, rr, pp)
    ok0, _ = feasible_map(kind, 0.0, rr, pp)
    res[kind] = (rr, pp, ok, ok0, L, poly)
    R, P = np.meshgrid(rr, pp)
    box = (R >= d["roll"][0]) & (R <= d["roll"][1]) & (P >= d["pitch"][0]) & (P <= d["pitch"][1])
    from matplotlib.path import Path
    inpoly = Path(poly).contains_points(np.stack([R.ravel(), P.ravel()], 1)).reshape(R.shape)
    print(f"\n{kind}: stroke {d['stroke'][0]*1000:.2f}-{d['stroke'][1]*1000:.2f} mm, margin {a.margin_mm:g} mm")
    print(f"  URDF box out of stroke: {100*(box & ~ok0).sum()/box.sum():.1f}% of its area"
          f" | polygon keeps {100*(inpoly & box).sum()/max((box & ok).sum(), 1):.1f}% of the feasible box,"
          f" stroke violations inside {v}/{nG}")
    print("  vertices [roll, pitch] rad (CCW): " + "  ".join(f"[{q[0]:+.3f}, {q[1]:+.3f}]" for q in poly))
    for side in ("R", "L"):
      for (s, sg, b), q0, nrm in cuts:
        sgn = 1.0 if side == "R" else -1.0  # left leg: roll mirrored
        n2 = np.array([sgn * nrm[0], nrm[1]])
        p1 = np.array([sgn * q0[0], q0[1]])
        p2 = p1 + 0.5 * np.array([n2[1], -n2[0]])  # (q - p1) x (p2 - p1) = -n.(q - p1)
        j1, j2 = (f"{side}_{x}" for x in names[kind])
        yaml.append(f"  - {{j1: {j1}, j2: {j2}, p1: [{p1[0]:.5f}, {p1[1]:.5f}], p2: [{p2[0]:.5f}, {p2[1]:.5f}]}}"
                    f"  # {'+y' if s > 0 else '-y'} cylinder {'min' if sg > 0 else 'max'}")
  open(f"{a.out}_compound.yaml", "w").write("\n".join(yaml) + "\n")
  print(f"\nmc_rtc descriptions: {a.out}_compound.yaml")
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
  for ax, kind in zip(axs, ("hip", "ankle")):
    rr, pp, ok, ok0, L, poly = res[kind]
    d = JOINTS[kind]
    lo, hi = d["stroke"]
    R, P = np.meshgrid(rr, pp)
    ax.contourf(R, P, ok0.astype(float), levels=[0.5, 1.5], colors=["#d9ead3"])
    for c, ls in ((0, "-"), (1, "--")):
      ax.contour(R, P, L[..., c] * 1000, levels=[lo * 1000, hi * 1000], colors="0.35", linestyles=ls, linewidths=0.8)
    ax.add_patch(plt.Rectangle((d["roll"][0], d["pitch"][0]), d["roll"][1] - d["roll"][0], d["pitch"][1] - d["pitch"][0],
                               fill=False, ec="#d62728", lw=1.3, label="URDF limits"))
    pc = np.vstack([poly, poly[:1]])
    ax.plot(pc[:, 0], pc[:, 1], "-", color="#1f77b4", lw=1.6, label=f"polygon ({a.margin_mm:g} mm margin)")
    ax.set_xlim(d["roll"][0] - 0.35, d["roll"][1] + 0.35)
    ax.set_ylim(d["pitch"][0] - 0.35, d["pitch"][1] + 0.35)
    ax.set_title(kind.capitalize())
    ax.set_xlabel("roll [rad]"); ax.set_ylabel("pitch [rad]")
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right", frameon=False)
  fig.tight_layout()
  fig.savefig(f"{a.out}.png", dpi=130)
  print(f"plot: {a.out}.png")


if __name__ == "__main__":
  main()
