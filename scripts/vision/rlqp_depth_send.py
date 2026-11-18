#!/usr/bin/env python3
"""rlqp_depth_send.py: Azure Kinect depth on the Jetson (rhps1v) -> rhps1_vision.py --source udp on the control PC.

Subscribes to the local Azure_Kinect_ROS_Driver (depth/image_raw 16UC1, depth/camera_info) and sends one pixel in
STRIDE per axis as UDP datagrams, each self-contained (frame seq, capture stamp, rows) so a lost datagram only loses
its rows. The stamp is the driver's header stamp: exposure start in the Jetson's clock (NTP-synced with rhps1c).
Intrinsics go out once a second in their own datagram.

  "RLD1" | u32 seq | f64 stamp | u16 w | u16 h | u16 row0 | u16 nrows | u8 stride | u8 pad | u16 pad | u16 mm[nrows*w]
  "RLK1" | u32 seq | u16 W | u16 H | u8 stride | u8 pad | u16 pad | f64 fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6

Run through depth_to_rhps1c.sh (starts the driver on a localhost-only ROS domain first).
"""
import argparse
import socket
import struct
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

FRAME = struct.Struct("<4sIdHHHHBBH")  # 28 bytes
INTR = struct.Struct("<4sIHHBBH12d")
assert FRAME.size == 28


class Sender(Node):
  def __init__(self, args):
    super().__init__("rlqp_depth_send")
    host, port = args.dest.rsplit(":", 1)
    self.addr = (socket.gethostbyname(host), int(port))
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.stride, self.min_dt = args.stride, 1.0 / args.max_rate
    self.rows = max(1, (1472 - FRAME.size) // (2 * ((512 + args.stride - 1) // args.stride)))
    self.intr, self.last_intr, self.last_send, self.seq, self.n = None, 0.0, 0.0, 0, 0
    self.t_report, self.lag = time.time(), []
    self.create_subscription(CameraInfo, "depth/camera_info", self.on_info, 1)
    self.create_subscription(Image, "depth/image_raw", self.on_image, qos_profile_sensor_data)
    self.get_logger().info(f"sending depth to {self.addr[0]}:{self.addr[1]}, stride {self.stride}, "
                           f"<= {args.max_rate:.0f} Hz")

  def on_info(self, m):
    if m.distortion_model != "rational_polynomial" or len(m.d) != 8:
      self.get_logger().error(f"unexpected distortion model {m.distortion_model} ({len(m.d)} coefficients)")
      return
    k = m.k
    self.intr = (m.width, m.height, k[0], k[4], k[2], k[5], *m.d)

  def on_image(self, m):
    now = time.time()
    if self.intr is None or m.encoding not in ("16UC1", "mono16") or now - self.last_send < 0.7 * self.min_dt:  # driver jitter: a 15 Hz stream at 15 Hz
      return
    self.last_send = now
    if now - self.last_intr > 1.0:
      W, H, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = self.intr
      self.sock.sendto(INTR.pack(b"RLK1", self.seq, W, H, self.stride, 0, 0, fx, fy, cx, cy,
                                 k1, k2, p1, p2, k3, k4, k5, k6), self.addr)
      self.last_intr = now
    img = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
    sub = np.ascontiguousarray(img[:: self.stride, :: self.stride])
    stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
    if abs(now - stamp) > 1.0:  # clock not what the driver thinks: fall back on arrival time
      stamp = now - 0.04
    self.lag.append(now - stamp)
    self.seq += 1
    h, w = sub.shape
    for r0 in range(0, h, self.rows):
      blk = sub[r0 : r0 + self.rows]
      self.sock.sendto(FRAME.pack(b"RLD1", self.seq, stamp, w, h, r0, blk.shape[0], self.stride, 0, 0)
                       + blk.tobytes(), self.addr)
    self.n += 1
    if now - self.t_report > 10.0:
      lag = np.array(self.lag)
      self.get_logger().info(f"{self.n} frames, {w}x{h}, exposure-to-send {np.median(lag) * 1e3:.0f} ms median, "
                             f"{lag.max() * 1e3:.0f} max")
      self.t_report, self.lag = now, []


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dest", default="10.16.236.199:5263", help="rhps1c, rhps1_vision.py --udp-port")
  ap.add_argument("--stride", type=int, default=4, help="keep one pixel in N per axis (WFOV binned: 512 -> 128)")
  ap.add_argument("--max-rate", type=float, default=15.0, help="frames per second sent")
  args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
  rclpy.init()
  node = Sender(args)
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
