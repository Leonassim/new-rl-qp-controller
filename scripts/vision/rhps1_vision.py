#!/usr/bin/env python3
"""rhps1_vision.py: onboard height map for the stairs vision policy (rl_controller obs_format 12).

Reads the controller's poses from /dev/shm/rhps1_vision_pose, turns depth frames into a 5 cm height map in the
controller's estimated world frame and writes it to /dev/shm/rhps1_vision_map (layout: src/vision_link.h). Same
map model as training (mjlab tasks/tracking/config/rhps1/vision_map.py): 48 x 48 cells around the pelvis, new
frame weight ALPHA on known cells, loaded soles written under the feet, cells never seen left unknown (NaN).

  --source sim     simulated depth camera (mc_mujoco): rays cast from the TRUE head pose against the stairs of
                   RLQP_STAIRS, points placed with the ESTIMATED pose, as on the robot; latency --lat.
  --source azure   Azure Kinect through pyk4a (WFOV 2x2 binned). Not tested on hardware yet.
  --source udp     depth from the Jetson (rlqp_depth_send.py, UDP --udp-port): each frame is placed with the head
                   pose at its exposure stamp (pose history kept in wall-clock time, both PCs NTP-synced), and the
                   map is stamped when it is written, so vision_max_age checks that the stream is alive while the
                   Wi-Fi delay (printed) only makes the newest terrain arrive later.

Example (simulation): RLQP_STAIRS="1.64 0 0 0.05 0.27 3 1.6" python3 rhps1_vision.py --source sim
"""
import argparse
import math
import mmap
import os
import struct
import socket
import sys
import threading
import time
import warnings

import numpy as np

POSE_SHM, MAP_SHM, CTL_SHM = "/dev/shm/rhps1_vision_pose", "/dev/shm/rhps1_vision_map", "/dev/shm/rhps1_vision_ctl"
CTL_MAGIC = 0x314C5443
MAGIC, POSE_RING, MAP_N = 0x31535652, 64, 48
POSE_DT = np.dtype([("t", "f8"), ("pelvis", "f8", 7), ("head", "f8", 7), ("pelvis_true", "f8", 7),
                    ("soles", "f8", (2, 4)), ("loaded", "f8", 2)])
assert POSE_DT.itemsize == 256
MAP_HDR = struct.Struct("<IIQdd2d")  # magic, n, seq, t_frame, res, origin[2]
assert MAP_HDR.size == 48
GATE_TAIL = struct.Struct("<IffI")  # after h: gate_magic, gate_high, gate_valid, rec_frames (vision_link.h)
GATE_MAGIC, GATE_HIGH = 0x31544147, 0.6
FOOT = [(x, y) for x in (-0.08, -0.04, 0.0, 0.04, 0.08) for y in (-0.04, 0.0, 0.04)]


def quat_mat(q):
  w, x, y, z = q
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rpy_mat(r, p, y):
  cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def pose_rt(v):
  return quat_mat(v[3:7]), np.asarray(v[:3], float)


class PoseReader:
  def __init__(self):
    while not os.path.exists(POSE_SHM):
      print("waiting for", POSE_SHM, "(controller with an obs_format 12 policy)", flush=True)
      time.sleep(1.0)
    self.open()

  def open(self):
    with open(POSE_SHM, "rb") as f:
      self.mm = mmap.mmap(f.fileno(), 16 + POSE_RING * 256, access=mmap.ACCESS_READ)
      self.ino = os.fstat(f.fileno()).st_ino

  def reopen_if_replaced(self):
    """A restarted controller may unlink and recreate the segment: the old mapping then never moves again."""
    try:
      if os.stat(POSE_SHM).st_ino != self.ino:
        self.open()
        print("pose segment replaced (controller restart): reopened", flush=True)
    except FileNotFoundError:
      pass

  def latest(self, k=1):
    magic, ring, count = struct.unpack_from("<IIQ", self.mm, 0)
    if magic != MAGIC or count == 0:
      return []
    out = []
    for c in range(max(0, count - k), count):
      out.append(np.frombuffer(self.mm, POSE_DT, 1, 16 + (c % ring) * 256)[0].copy())
    if struct.unpack_from("<Q", self.mm, 8)[0] - (count - len(out)) > ring:  # lapped while copying
      return []
    return out


def gate_stats(pts, soles, n_rays):
  """The controller's vision gate inputs for one frame: share of its points GATE_HIGH above the lowest sole (a person,
  clutter at body height), share of the ROI rays that returned a point (camera blind)."""
  if n_rays <= 0:
    return 0.0, 0.0
  if len(pts) == 0:
    return 0.0, 0.0
  z0 = min(float(soles[0][2]), float(soles[1][2]))
  return float(np.mean(pts[:, 2] - z0 > GATE_HIGH)), min(1.0, len(pts) / n_rays)


def screen(pts, soles, n_rays, args):
  """A doubtful frame (the controller gate's thresholds) writes nothing; points over --clip-high above the lowest
  sole never reach the map (vision_map GATE_SKIP, CLIP). Returns the points to write and the gate inputs."""
  gs = gate_stats(pts, soles, n_rays) if n_rays else None
  if gs is not None and (gs[0] > args.gate_hfrac or gs[1] < args.gate_valid):
    return pts[:0], gs
  if args.clip_high > 0 and len(pts):
    z0 = min(float(soles[0][2]), float(soles[1][2]))
    pts = pts[pts[:, 2] - z0 <= args.clip_high]
  return pts, gs


class CtlReader:
  """GUI requests from the controller (vision_link.h CtlBlock); absent segment: no request."""

  def __init__(self):
    self.mm, self.ino = None, None

  def record(self):
    try:
      ino = os.stat(CTL_SHM).st_ino
      if ino != self.ino:
        with open(CTL_SHM, "rb") as f:
          self.mm = mmap.mmap(f.fileno(), 16, access=mmap.ACCESS_READ)
        self.ino = ino
      magic, rec = struct.unpack_from("<II", self.mm, 0)
      return magic == CTL_MAGIC and rec == 1
    except (FileNotFoundError, ValueError, OSError):
      return False


class MapWriter:
  def __init__(self):
    self.h_end = MAP_HDR.size + MAP_N * MAP_N * 4
    size = self.h_end + GATE_TAIL.size
    fd = os.open(MAP_SHM, os.O_RDWR | os.O_CREAT, 0o666)
    os.ftruncate(fd, size)
    self.mm = mmap.mmap(fd, size)
    os.close(fd)
    self.seq = 0
    MAP_HDR.pack_into(self.mm, 0, 0, MAP_N, 0, -1.0, 0.05, 0.0, 0.0)

  def write(self, t_frame, res, origin, h, gate=None, rec_frames=0):
    """gate: (share of the frame's points GATE_HIGH above the lowest sole, share of the ROI rays returning)."""
    self.seq += 1
    struct.pack_into("<Q", self.mm, 8, self.seq)  # odd: inside
    self.mm[MAP_HDR.size:self.h_end] = np.ascontiguousarray(h, np.float32).tobytes()
    if gate is not None:
      GATE_TAIL.pack_into(self.mm, self.h_end, GATE_MAGIC, gate[0], gate[1], rec_frames)
    MAP_HDR.pack_into(self.mm, 0, MAGIC, MAP_N, self.seq, t_frame, res, origin[0], origin[1])
    self.seq += 1
    struct.pack_into("<Q", self.mm, 8, self.seq)


class HeightMap:
  """Torus of n x n cells tagged with the absolute cell they hold (vision_map.VisionMap, one robot)."""

  def __init__(self, n=MAP_N, res=0.05, alpha=0.5):
    self.n, self.res, self.alpha = n, res, alpha
    self.h = np.zeros(n * n)
    self.key = np.full(n * n, -1, np.int64)
    self.centre = np.zeros(2)

  def _slots(self, xy):
    ix = np.floor(xy[:, 0] / self.res).astype(np.int64)
    iy = np.floor(xy[:, 1] / self.res).astype(np.int64)
    return ix, iy, (ix % self.n) * self.n + (iy % self.n), (ix + (1 << 20)) * (1 << 21) + (iy + (1 << 20))

  def write(self, pts, replace):
    if len(pts) == 0:
      return
    ix, iy, slot, key = self._slots(pts)
    cx, cy = np.floor(self.centre / self.res).astype(np.int64)
    half = self.n // 2
    ok = (np.abs(ix - cx) < half) & (np.abs(iy - cy) < half)
    slot, key, z = slot[ok], key[ok], pts[ok, 2]
    n2 = self.n * self.n
    s = np.bincount(slot, z, n2)
    c = np.bincount(slot, None, n2)
    hit = c > 0
    kb = np.full(n2, -1, np.int64)
    kb[slot] = key
    m = np.where(hit, s / np.maximum(c, 1), 0.0)
    if replace:
      self.h = np.where(hit, m, self.h)
    else:
      same = hit & (self.key == kb)
      self.h = np.where(same, (1 - self.alpha) * self.h + self.alpha * m, np.where(hit, m, self.h))
    self.key = np.where(hit, kb, self.key)

  def under(self, pts):
    """Known heights of the cells under pts (empty where never seen)."""
    _, _, slot, key = self._slots(pts)
    ok = self.key[slot] == key
    return self.h[slot[ok]]

  def grid(self):
    """Plain n x n grid around the centre: (origin xy, heights with NaN where unknown), cell (i, j) at i * n + j."""
    c0 = np.floor(self.centre / self.res).astype(np.int64) - self.n // 2
    i, j = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="ij")
    ax, ay = (c0[0] + i).ravel(), (c0[1] + j).ravel()
    slot = (ax % self.n) * self.n + (ay % self.n)
    key = (ax + (1 << 20)) * (1 << 21) + (ay + (1 << 20))
    h = np.where(self.key[slot] == key, self.h[slot], np.nan)
    return c0 * self.res, h


class Persist:
  """--persist: a cell's points reach the map only if the previous frame put that cell within --persist-tol of the
  same height. A person grazing the ROI edge stays one or two frames in a cell; a step stays."""

  def __init__(self, res, tol):
    self.res, self.tol, self.prev = res, tol, {}

  def __call__(self, pts):
    if self.tol <= 0 or len(pts) == 0:
      return pts
    ij = np.floor(pts[:, :2] / self.res).astype(np.int64)
    key = (ij[:, 0] + (1 << 20)) * (1 << 21) + (ij[:, 1] + (1 << 20))
    uk, inv = np.unique(key, return_inverse=True)
    mz = np.bincount(inv, pts[:, 2]) / np.bincount(inv)
    pz = np.array([self.prev.get(int(k), np.nan) for k in uk])
    keep = np.abs(mz - pz) <= self.tol
    self.prev = dict(zip(uk.tolist(), mz.tolist()))
    return pts[keep[inv]]


def stairs_height(x, y, g):
  cx, cy, yaw, hh, tread, nsteps, platform = g
  dx, dy = x - cx, y - cy
  c, s = math.cos(yaw), math.sin(yaw)
  d = np.maximum(np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy))
  k = nsteps - np.ceil((d - platform / 2) / tread)
  return np.where(d <= platform / 2, nsteps * hh, np.where(k > 0, k * hh, 0.0))


def boxes_from_env():
  """RLQP_BOXES "x0 x1 y0 y1 h; ...": axis-aligned blocks on the floor (mc_mujoco_stairs.sh box scenes)."""
  return [[float(v) for v in b.split()] for b in os.environ.get("RLQP_BOXES", "").split(";") if b.strip()]


class SimCamera:
  """Pinhole rays inside the 120 deg WFOV circle, below the horizon at the nominal head pitch, cast on RLQP_STAIRS
  and RLQP_BOXES."""

  def __init__(self, args):
    g = os.environ.get("RLQP_STAIRS", "")
    self.geom = [float(v) for v in g.split()] if g else None
    self.boxes = boxes_from_env()
    if self.geom is None and not self.boxes:
      print("RLQP_STAIRS and RLQP_BOXES unset: flat ground", flush=True)
    t = math.tan(math.radians(60.0))
    u = np.linspace(-t, t, args.sim_rays)
    gu, gv = np.meshgrid(u, u)
    d = np.stack([gu.ravel(), gv.ravel(), -np.ones(gu.size)], 1)  # MuJoCo camera axes: -z forward, y up
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] <= -math.cos(math.radians(60.0))]
    self.dirs = d
    self.rng = np.random.default_rng(0)
    self.noise = args.noise

  def height(self, x, y):
    h = stairs_height(x, y, self.geom) if self.geom else np.zeros(np.shape(x))
    for x0, x1, y0, y1, bh in self.boxes:
      h = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), np.maximum(h, bh), h)
    return h

  def points(self, R_cam, t_cam):
    """Hit points in the camera frame (m), rays that hit within 0.25-2.88 m."""
    dw = self.dirs @ R_cam.T
    dw = dw[dw[:, 2] < -0.05]
    ts = np.arange(0.25, 2.88, 0.01)
    p = t_cam[None, None, :] + ts[None, :, None] * dw[:, None, :]
    terr = self.height(p[..., 0], p[..., 1])
    below = p[..., 2] <= terr
    hit = below.any(1)
    k = np.argmax(below, 1)[hit]
    dw = dw[hit]
    lo, hi = ts[np.maximum(k - 1, 0)], ts[k]
    for _ in range(6):  # bisection on the step that crossed the surface
      mid = 0.5 * (lo + hi)
      q = t_cam + mid[:, None] * dw
      under = q[:, 2] <= self.height(q[:, 0], q[:, 1])
      hi, lo = np.where(under, mid, hi), np.where(under, lo, mid)
    r = hi + self.rng.normal(0.0, self.noise, hi.shape)
    return (r[:, None] * dw) @ R_cam  # back in the camera frame


class AzureCamera:
  """Azure Kinect, depth WFOV 2x2 binned, through pyk4a. Points in the MuJoCo camera convention."""

  def __init__(self, args):
    import pyk4a  # noqa: PLC0415
    from pyk4a import Config, DepthMode, FPS, ColorResolution  # noqa: PLC0415

    self.k4a = pyk4a.PyK4A(Config(color_resolution=ColorResolution.OFF, depth_mode=DepthMode.WFOV_2X2BINNED,
                                  camera_fps=FPS.FPS_30, synchronized_images_only=False))
    self.k4a.start()
    self.step = args.azure_stride

  def points(self):
    cap = self.k4a.get_capture()
    pc = cap.depth_point_cloud  # H x W x 3, mm, x right, y down, z forward
    if pc is None:
      return None
    p = pc[:: self.step, :: self.step].reshape(-1, 3).astype(np.float64) / 1000.0
    p = p[(p[:, 2] > 0.25) & (p[:, 2] < 2.88)]
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], 1)  # to x right, y up, -z forward


def ground_mask(rays, cam_R, cam_t, head_in_pelvis, args):
  """Pixels whose ray lands on the training ROI at the nominal stance (vision_map.DepthCamPatternCfg): head at
  --head-pitch-deg, ground SCAN_OFFSET below the pelvis. Rays stay fixed in HEAD_P_LINK, as in training, so the
  ceiling and the walls never reach the map whatever the head does."""
  x, y = rays
  v = np.stack([x, -y, -np.ones_like(x)], -1)
  v /= np.linalg.norm(v, axis=-1, keepdims=True)
  R_ph = rpy_mat(0.0, math.radians(args.head_pitch_deg), 0.0)
  d = v @ (R_ph @ cam_R).T
  c = head_in_pelvis + R_ph @ cam_t
  with np.errstate(all="ignore"):
    s = (-args.scan_offset - c[2]) / d[..., 2]
    hx, hy = c[0] + s * d[..., 0], c[1] + s * d[..., 1]
  rx0, rx1 = (float(a) for a in args.roi_x.split(","))
  m = args.roi_margin
  keep = (d[..., 2] < 0) & (hx >= rx0 - m) & (hx <= rx1 + m) & (np.abs(hy) <= args.roi_y + m)
  return keep & (-v[..., 2] >= math.cos(math.radians(args.cone_deg)))


def unproject_table(W, H, stride, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6):
  """Normalised rays (x/z, y/z) of the strided pixels for the rational polynomial model, NaN where the inversion does
  not converge (past the lens' metric radius)."""
  u, v = np.meshgrid(np.arange(0, W, stride, dtype=float), np.arange(0, H, stride, dtype=float))
  xd, yd = (u - cx) / fx, (v - cy) / fy

  def distort(x, y):
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

  x, y = xd.copy(), yd.copy()
  for _ in range(60):
    dx, dy = distort(x, y)
    x, y = x + (xd - dx), y + (yd - dy)
  dx, dy = distort(x, y)
  bad = (np.hypot(dx - xd, dy - yd) > 1e-6) | ~np.isfinite(x) | ~np.isfinite(y)
  x[bad], y[bad] = np.nan, np.nan
  return x, y


class UdpDepthCamera:
  """Frames from rlqp_depth_send.py. A thread assembles datagrams; frames() hands over the finished ones."""

  FRAME = struct.Struct("<4sIdHHHHBBH")
  INTR = struct.Struct("<4sIHHBBH12d")

  def __init__(self, args):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    self.sock.bind(("0.0.0.0", args.udp_port))
    self.edge = args.edge_filter
    self.lock, self.done, self.cur, self.intr, self.rays = threading.Lock(), [], None, None, None
    self.n_lost = 0
    threading.Thread(target=self._rx, daemon=True).start()

  def _finish(self):
    f = self.cur
    if f is not None:
      if f["rows"] >= 0.8 * f["d"].shape[0]:
        self.done.append((f["stamp"], f["d"]))
        del self.done[:-4]
      else:
        self.n_lost += 1
    self.cur = None

  def _rx(self):
    while True:
      p, _ = self.sock.recvfrom(2048)
      if p[:4] == b"RLK1" and len(p) == self.INTR.size:
        v = self.INTR.unpack(p)
        intr = (v[2], v[3], v[4]) + v[7:]
        if intr != self.intr:
          rays = unproject_table(*intr)
          with self.lock:
            self.intr, self.rays = intr, rays
          print(f"udp: intrinsics {v[2]}x{v[3]} stride {v[4]}, fx {v[7]:.1f}, "
                f"{np.isfinite(rays[0]).mean():.0%} of the pixels unprojected", flush=True)
        continue
      if p[:4] != b"RLD1" or len(p) < self.FRAME.size:
        continue
      _, seq, stamp, w, h, r0, nr, _, _, _ = self.FRAME.unpack_from(p)
      with self.lock:
        if self.cur is not None and seq != self.cur["seq"]:
          if seq < self.cur["seq"]:
            continue
          self._finish()
        if self.cur is None:
          self.cur = {"seq": seq, "stamp": stamp, "rows": 0, "d": np.zeros((h, w), np.uint16)}
        self.cur["d"][r0 : r0 + nr] = np.frombuffer(p, np.uint16, nr * w, self.FRAME.size).reshape(nr, w)
        self.cur["rows"] += nr
        if self.cur["rows"] >= h:
          self._finish()

  def frames(self):
    with self.lock:
      out, self.done = self.done, []
      rays = self.rays
    return out, rays

  def points(self, d_mm, rays, mask=None):
    """Points in the MuJoCo camera convention (x right, y up, -z forward), same as AzureCamera."""
    d = d_mm.astype(np.float64) / 1000.0
    d[d_mm == 0] = np.nan
    if mask is not None:
      d[~mask] = np.nan
    if self.edge > 0:  # flying pixels between a step edge and what lies behind it
      pad = np.pad(d, 1, constant_values=np.nan)
      nb = np.stack([pad[i : i + d.shape[0], j : j + d.shape[1]] for i in range(3) for j in range(3)])
      with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods at the lens edge
        med = np.nanmedian(nb, axis=0)
      d[np.abs(d - med) > np.maximum(self.edge, self.edge * d)] = np.nan
    x, y = rays
    p = np.stack([x * d, y * d, d], -1).reshape(-1, 3)
    r = np.linalg.norm(p, axis=1)
    p = p[np.isfinite(r) & (r > 0.25) & (r < 2.88)]
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], 1)


class RosViz:
  """--ros: known map cells and the newest frame's points as PointCloud2 in the controller's world frame (mc_rtc's
  robot_map), for RViz. The map carries the vision height correction (leg odometry), so it can sit a few cm off the
  real robot model in z."""

  def __init__(self, frame, rate):
    import rclpy  # noqa: PLC0415
    from sensor_msgs.msg import PointCloud2  # noqa: PLC0415
    from sensor_msgs_py import point_cloud2  # noqa: PLC0415
    from std_msgs.msg import Header  # noqa: PLC0415

    rclpy.init()
    self.node = rclpy.create_node("rhps1_vision")
    self.pub_map = self.node.create_publisher(PointCloud2, "rhps1_vision/height_map", 1)
    self.pub_pts = self.node.create_publisher(PointCloud2, "rhps1_vision/points", 1)
    self.pc2, self.Header, self.frame, self.dt, self.last = point_cloud2, Header, frame, 1.0 / rate, 0.0

  def publish(self, origin, h, res, pts):
    now = time.time()
    if now - self.last < self.dt:
      return
    self.last = now
    hdr = self.Header()
    hdr.frame_id = self.frame
    hdr.stamp = self.node.get_clock().now().to_msg()
    n = int(round(np.sqrt(h.size)))
    h = h.reshape(n, n)
    i, j = np.nonzero(np.isfinite(h))
    cells = np.stack([origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res, h[i, j]], 1)
    self.pub_map.publish(self.pc2.create_cloud_xyz32(hdr, cells.astype(np.float32)))
    if pts is not None and len(pts):
      self.pub_pts.publish(self.pc2.create_cloud_xyz32(hdr, pts[:: max(1, len(pts) // 4000)].astype(np.float32)))


def save_part(prefix, k, rec, rays, mask, cam_t, args):
  np.savez(f"{prefix}_{k:03d}.npz", **{key: np.array(v) for key, v in rec.items()}, rays_x=rays[0], rays_y=rays[1],
           mask=mask if mask is not None else np.isfinite(rays[0]), cam_xyz=cam_t,
           cam_rpy=np.array([float(v) for v in args.cam_rpy.split(",")]))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--source", choices=("sim", "azure", "udp"), default="sim")
  ap.add_argument("--udp-port", type=int, default=5263, help="udp: port rlqp_depth_send.py sends to")
  ap.add_argument("--edge-filter", type=float, default=0.05,
                  help="udp: drop pixels off their 3x3 median by more than this fraction of depth (and m), 0 = off")
  ap.add_argument("--ros", action="store_true", help="publish rhps1_vision/height_map and /points for RViz")
  ap.add_argument("--ros-frame", default="robot_map")
  ap.add_argument("--record", default="", help="udp: save every frame (depth, stamp, matched poses) to this .npz")
  ap.add_argument("--record-dir", default=os.path.expanduser("~/rl_deploy/logs"),
                  help="udp: where the controller GUI's 'Record depth' writes vision_<date>_NNN.npz")
  ap.add_argument("--rate", type=float, default=25.0, help="depth frames per second of controller time")
  ap.add_argument("--lat", type=float, default=0.08, help="sim: frame-to-map latency, s of controller time")
  ap.add_argument("--azure-lat", type=float, default=0.04, help="azure: capture-to-host delay assumed, s")
  ap.add_argument("--noise", type=float, default=0.01, help="sim: range noise, m")
  ap.add_argument("--sim-rays", type=int, default=72, help="sim: rays per image side")
  ap.add_argument("--azure-stride", type=int, default=4, help="azure: keep one pixel in N per axis")
  ap.add_argument("--alpha", type=float, default=0.5)
  ap.add_argument("--align", action="store_true",
                  help="at each touchdown flat on one surface, shift the whole map so the sole sits on the height the camera measured there (removes the estimator's height drift since the step was seen)")
  ap.add_argument("--roi-x", default="0.0,1.6", help="udp: ground ROI ahead of the pelvis, as vision_map ROI_X")
  ap.add_argument("--roi-y", type=float, default=0.6)
  ap.add_argument("--roi-margin", type=float, default=0.05)
  ap.add_argument("--cone-deg", type=float, default=60.0)
  ap.add_argument("--head-pitch-deg", type=float, default=25.0, help="HEAD_P the ROI pixels are aimed with")
  ap.add_argument("--scan-offset", type=float, default=0.84, help="ground below the pelvis, stairs SCAN_OFFSET")
  ap.add_argument("--all-pixels", action="store_true", help="udp: no ROI mask (ceiling and walls reach the map)")
  ap.add_argument("--gate-hfrac", type=float, default=0.08, help="controller vision_gate_hfrac: over it, not written")
  ap.add_argument("--gate-valid", type=float, default=0.1, help="controller vision_gate_valid: under it, not written")
  ap.add_argument("--persist-tol", type=float, default=0.05,
                  help="m: a cell is written only if the previous frame saw it this close in height; 0: off")
  ap.add_argument("--clip-high", type=float, default=0.6, help="m above the lowest sole never written; 0: off")
  ap.add_argument("--cam-xyz", default="0.1190,0.0017,0.1825",
                  help="depth camera in HEAD_P_LINK, m (fitted on rhps1, 2026-11-18; vision_map.py CAM_POS)")
  ap.add_argument("--cam-rpy", default="1.49513,-0.00503,-1.57095", help="depth camera in HEAD_P_LINK, URDF rpy")
  args = ap.parse_args()

  cam_t = np.array([float(v) for v in args.cam_xyz.split(",")])
  cam_R = rpy_mat(*[float(v) for v in args.cam_rpy.split(",")])
  poses, out, hm, ctl = PoseReader(), MapWriter(), HeightMap(alpha=args.alpha), CtlReader()
  rec_prefix, rec_n, t_ctl, t_pose = args.record.removesuffix(".npz"), 0, 0.0, time.time()
  persist = Persist(hm.res, args.persist_tol)
  cam = {"sim": SimCamera, "azure": AzureCamera, "udp": UdpDepthCamera}[args.source](args)
  hist, lat_net, rec = [], [], {"t": [], "stamp": [], "depth": [], "head": [], "pelvis": []}
  viz = RosViz(args.ros_frame, 5.0) if args.ros else None
  foot = np.array(FOOT)
  pending, last_frame, last_t, n_frames, n_skip, t_wall = [], -1e9, -1.0, 0, 0, time.time()
  was_loaded, shifts = [False, False], []
  mask, mask_rays = None, None
  print(f"vision: source {args.source}, {args.rate:.0f} Hz, writing {MAP_SHM}", flush=True)
  while True:
    lp = poses.latest(1)
    if not lp or lp[0]["t"] == last_t:
      if time.time() - t_pose > 2.0:
        poses.reopen_if_replaced()
        t_pose = time.time()
      time.sleep(0.002)
      continue
    t_pose = time.time()
    if not args.record and time.time() - t_ctl > 0.5:  # GUI record checkbox
      t_ctl = time.time()
      want = ctl.record()
      if want and not rec_prefix:
        rec_prefix, rec_n = os.path.join(args.record_dir, time.strftime("vision_%Y%m%d_%H%M%S")), 0
        os.makedirs(args.record_dir, exist_ok=True)
        print(f"recording to {rec_prefix}_NNN.npz", flush=True)
      elif not want and rec_prefix:
        if rec["t"] and rays is not None:
          save_part(rec_prefix, rec_n, rec, rays, mask, cam_t, args)
        print(f"recording stopped: {rec_prefix}, {rec_n * 150 + len(rec['t'])} frames", flush=True)
        rec_prefix, rec = "", {key: [] for key in rec}
    pz = lp[0]
    last_t = float(pz["t"])
    hist.append((time.time(), pz))
    del hist[: max(0, len(hist) - 800)]  # 4 s at 200 Hz
    if last_t < last_frame:  # controller restarted
      hm, pending, last_frame = HeightMap(alpha=args.alpha), [], -1e9
    R_pe, t_pe = pose_rt(pz["pelvis"])
    hm.centre = t_pe[:2]
    for k in range(2):  # loaded soles: the kinematic footprint, as training writes it
      loaded = pz["loaded"][k] > 0.5
      if loaded:
        x, y, z, yaw = pz["soles"][k]
        c, s = math.cos(yaw), math.sin(yaw)
        fp = np.stack([x + c * foot[:, 0] - s * foot[:, 1], y + s * foot[:, 0] + c * foot[:, 1],
                       np.full(len(foot), z)], 1)
        if args.align and not was_loaded[k]:
          hu = hm.under(fp)
          if len(hu) >= 8 and np.ptp(hu) < 0.02:  # fully on one surface the camera has seen
            dz = float(np.clip(z - np.median(hu), -0.05, 0.05))
            hm.h += dz
            shifts.append(dz)
        hm.write(fp, replace=True)
      was_loaded[k] = loaded
    if args.source == "udp":
      frames, rays = cam.frames()
      if rays is not None and not args.all_pixels and mask_rays is not rays:
        head_in_pelvis = R_pe.T @ (pose_rt(pz["head"])[1] - t_pe)
        mask, mask_rays = ground_mask(rays, cam_R, cam_t, head_in_pelvis, args), rays
        print(f"udp: {mask.sum()} of {np.isfinite(rays[0]).sum()} pixels aimed at the ROI (head at "
              f"{args.head_pitch_deg:.0f} deg)", flush=True)
      for stamp, d in frames:
        if rays is None or not hist or stamp < hist[0][0]:
          continue
        k = min(range(len(hist)), key=lambda i: abs(hist[i][0] - stamp))
        R_hc, t_hc = pose_rt(hist[k][1]["head"])
        pc = cam.points(d, rays, mask)
        pw = pc @ (R_hc @ cam_R).T + (R_hc @ cam_t + t_hc)
        n_rays = int(mask.sum()) if mask is not None else int(np.isfinite(rays[0]).sum())
        pk, gs = screen(pw, hist[k][1]["soles"], n_rays, args)
        pk = persist(pk)
        n_skip += len(pk) == 0
        hm.write(pk, replace=False)
        origin, h = hm.grid()
        # stamped now: the controller's gate watches the stream, not the Wi-Fi delay
        out.write(last_t, hm.res, origin, h, gs, rec_n * 150 + len(rec["t"]) if rec_prefix else 0)
        if viz:
          viz.publish(origin, h, hm.res, pw)
        lat_net.append(time.time() - stamp)
        n_frames += 1
        if rec_prefix:
          for key, val in (("t", hist[k][1]["t"]), ("stamp", stamp), ("depth", d), ("head", hist[k][1]["head"]),
                           ("pelvis", hist[k][1]["pelvis"])):
            rec[key].append(val)
          if len(rec["t"]) >= 150:  # 10 s chunks: rewriting one growing file loaded the control PC's disk
            save_part(rec_prefix, rec_n, rec, rays, mask, cam_t, args)
            rec_n += 1
            rec = {key: [] for key in rec}
    elif last_t - last_frame >= 1.0 / args.rate:
      last_frame = last_t
      R_he, t_he = pose_rt(pz["head"])
      if args.source == "sim":
        if np.isnan(pz["pelvis_true"][0]):
          print("no true pose in the pose block: --source sim needs mc_mujoco", flush=True)
          sys.exit(1)
        R_pt, t_pt = pose_rt(pz["pelvis_true"])
        R_ht = R_pt @ R_pe.T @ R_he  # head relative to the pelvis from the joints, pelvis from the physics
        t_ht = R_pt @ (R_pe.T @ (t_he - t_pe)) + t_pt
        pc = cam.points(R_ht @ cam_R, R_ht @ cam_t + t_ht)
        t_cap = last_t
      else:
        pc = cam.points()
        t_cap = last_t - args.azure_lat
        if pc is None:
          continue
      pts = pc @ (R_he @ cam_R).T + (R_he @ cam_t + t_he)  # estimated frame, as the robot knows it
      n_rays = len(cam.dirs) if args.source == "sim" else None
      pk, gs = screen(pts, pz["soles"], n_rays, args)
      pk = persist(pk)
      n_skip += len(pk) == 0
      pending.append((t_cap, pk, gs))
      n_frames += 1
    lat = args.lat if args.source == "sim" else 0.0
    while pending and pending[0][0] + lat <= last_t:
      t_cap, pts, gs = pending.pop(0)
      hm.write(pts, replace=False)
      origin, h = hm.grid()
      out.write(t_cap, hm.res, origin, h, gs)
      if viz:
        viz.publish(origin, h, hm.res, pts)
    if time.time() - t_wall > 5.0:
      t_wall = time.time()
      _, h = hm.grid()
      al = f" | align {len(shifts)} shifts, mean {np.mean(shifts) * 100:+.1f} cm, max |{np.abs(shifts).max() * 100:.1f}| cm" if shifts else ""
      net = (f" | exposure->map {np.median(lat_net) * 1e3:.0f} ms median, {np.max(lat_net) * 1e3:.0f} max,"
             f" {cam.n_lost} frames lost") if lat_net else ""
      lat_net = []
      print(f"t {last_t:7.2f} s | {n_frames} frames, {n_skip} doubtful | known {np.isfinite(h).mean():.0%}{al}{net}", flush=True)


if __name__ == "__main__":
  main()
