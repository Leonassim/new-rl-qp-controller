#!/bin/bash
# Gamepad on rhps1t -> NewRLQPController on rhps1c (UDP 5262, joystick_udp_port in rhps1c:~/rl_deploy/mc_rtc.yaml).
# Picks the first joystick with 4 axes or more (js0 here is a 2-axis HID dongle). Ctrl-C or unplugging the pad stops
# the stream; the controller then ramps the command to zero after 0.5 s.
exec python3 -u "$(dirname "$(readlink -f "$0")")/rlqp_joy.py" send --udp 10.16.236.199:5262 "$@"
