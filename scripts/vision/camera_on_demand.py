#!/usr/bin/env python3
"""camera_on_demand.py: on rhps1v, runs depth_to_rhps1c.sh only while rhps1c asks for the camera.

rhps1_vision.py on rhps1c sends b"rhps1_camera_on" once a second while the controller wants the Azure (vision slot
loaded or depth recording). The camera starts on the first wake-up and stops OFF_S after the last one.
"""

import os
import signal
import socket
import subprocess
import time

PORT = int(os.environ.get("RLQP_CAMERA_PORT", "5264"))
OFF_S = float(os.environ.get("RLQP_CAMERA_OFF_S", "10"))
CMD = os.path.join(os.path.dirname(os.path.realpath(__file__)), "depth_to_rhps1c.sh")


def stop(proc):
  try:
    os.killpg(proc.pid, signal.SIGINT)  # depth_to_rhps1c.sh's trap stops the driver too
    proc.wait(timeout=10)
  except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
  except ProcessLookupError:
    pass


def main():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.bind(("0.0.0.0", PORT))
  s.settimeout(1.0)
  print(f"camera on demand: UDP {PORT}, {CMD}, off {OFF_S:.0f} s after the last wake-up", flush=True)
  proc, last = None, -1e9
  try:
    while True:
      try:
        data, addr = s.recvfrom(64)
        if data == b"rhps1_camera_on":
          last = time.time()
      except socket.timeout:
        pass
      want = time.time() - last < OFF_S
      if proc is not None and proc.poll() is not None:
        print(f"{time.strftime('%T')} camera exited ({proc.returncode})", flush=True)
        proc = None
      if want and proc is None:
        print(f"{time.strftime('%T')} camera on (asked by {addr[0]})", flush=True)
        proc = subprocess.Popen([CMD], start_new_session=True)
      elif not want and proc is not None:
        print(f"{time.strftime('%T')} camera off (no request for {OFF_S:.0f} s)", flush=True)
        stop(proc)
        proc = None
  finally:
    if proc is not None:
      stop(proc)


if __name__ == "__main__":
  main()
