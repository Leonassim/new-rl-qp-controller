#!/usr/bin/env python3
"""Gamepad -> NewRLQPController on another PC (global config key joystick_udp_port, src/joystick_udp.h).

  send  [--device auto|/dev/input/jsN] (--udp HOST:PORT | --stdout)
        reads the Linux joystick device, sends its raw state at 50 Hz and on every change. Unplugged pad: nothing is
        sent, so the controller ramps the command to zero after joystick_udp_timeout. auto: the first js device with
        at least 4 axes (rhps1t's js0 is a 2-axis HID dongle, not a gamepad).
  relay PORT
        reads packets on stdin, forwards them to 127.0.0.1:PORT. Runs on the control PC, fed through ssh.

Same LAN (rhps1t):  rlqp_joy.py send --udp 10.16.236.199:5262
Through the VPN (tardis, TCP only):
  rlqp_joy.py send --stdout | ssh rhps1user@rhps1c.s.aist.go.jp \\
      python3 -u rl_deploy/src/rl_controller/scripts/joystick/rlqp_joy.py relay 5262
"""
import argparse
import fcntl
import glob
import os
import select
import socket
import struct
import sys
import time

PACKET = struct.Struct("<4sIBBH8hHH")  # 32 bytes, see joystick_udp.h
MAGIC = b"RLJ1"
JS_EVENT = struct.Struct("IhBB")
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80


def log(*a):
    print("[rlqp_joy]", *a, file=sys.stderr, flush=True)


JSIOCGAXES = 0x80016A11


def axes_count(dev):
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return 0
    try:
        buf = bytearray(1)
        fcntl.ioctl(fd, JSIOCGAXES, buf)
        return buf[0]
    except OSError:
        return 0
    finally:
        os.close(fd)


def find_gamepad():
    devs = sorted(glob.glob("/dev/input/js*"), key=lambda d: int(d[13:]) if d[13:].isdigit() else 99)
    return next((d for d in devs if axes_count(d) >= 4), None)


def device_name(dev):
    try:
        with open(f"/sys/class/input/{os.path.basename(dev)}/device/name") as f:
            return f.read().strip()
    except OSError:
        return "?"


def send(args):
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        addr = (socket.gethostbyname(host), int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        out = lambda p: sock.sendto(p, addr)
    else:
        out = lambda p: (sys.stdout.buffer.write(p), sys.stdout.buffer.flush())
    seq, waiting = 0, False
    while True:
        dev = find_gamepad() if args.device == "auto" else args.device
        try:
            if dev is None:
                raise FileNotFoundError(2, "no joystick with 4 axes or more")
            fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            if not waiting:
                log(f"{dev or args.device}: {e.strerror}, waiting for a gamepad")
            waiting = True
            time.sleep(1.0)
            continue
        waiting = False
        axes, buttons, n_axes, n_buttons = [0] * 8, 0, 0, 0
        log(f"{dev}: {device_name(dev)}")
        reported, last = False, 0.0
        try:
            while True:
                changed = False
                r, _, _ = select.select([fd], [], [], 0.02)
                if r:
                    data = os.read(fd, JS_EVENT.size * 64)
                    for i in range(0, len(data) - JS_EVENT.size + 1, JS_EVENT.size):
                        _, value, kind, number = JS_EVENT.unpack_from(data, i)
                        if kind & JS_AXIS and number < 8:
                            axes[number] = value
                            n_axes = max(n_axes, number + 1)
                        elif kind & JS_BUTTON and number < 16:
                            buttons = buttons | (1 << number) if value else buttons & ~(1 << number)
                            n_buttons = max(n_buttons, number + 1)
                        changed = True
                now = time.time()
                if not reported and now - last > 0.2 and (n_axes or n_buttons):
                    log(f"{n_axes} axes, {n_buttons} buttons")
                    reported = True
                if changed or now - last >= 0.02:
                    # Wall-clock milliseconds: increasing across sender restarts, which the receiver relies on.
                    seq = max(seq + 1, int(now * 1000) & 0xFFFFFFFF)
                    out(PACKET.pack(MAGIC, seq, n_axes, n_buttons, 0, *axes, buttons, 0))
                    last = now
        except OSError as e:
            log(f"{dev}: {e.strerror}, stopped sending")
            os.close(fd)
            time.sleep(1.0)


def relay(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", args.port)
    stdin = sys.stdin.buffer
    n = 0
    log(f"relaying to {addr[0]}:{addr[1]}")
    while True:
        p = stdin.read(PACKET.size)
        if len(p) < PACKET.size:
            log(f"input closed after {n} packets")
            return
        if p[:4] != MAGIC:
            log("bad packet, input out of sync")
            return
        sock.sendto(p, addr)
        n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--device", default="auto")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--udp", metavar="HOST:PORT")
    g.add_argument("--stdout", action="store_true")
    r = sub.add_parser("relay")
    r.add_argument("port", type=int)
    args = ap.parse_args()
    try:
        send(args) if args.cmd == "send" else relay(args)
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
