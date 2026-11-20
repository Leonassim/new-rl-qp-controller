#!/bin/bash
# vision_service.sh {rhps1v|rhps1c}: keeps the vision side running from boot, restarted if it exits.
#   rhps1v: camera_on_demand.py, the Azure (depth_to_rhps1c.sh) only while rhps1c asks for it
#   rhps1c: height map -> controller shared memory + RViz (vision_rhps1c.sh), wakes the Jetson's camera while the
#           controller wants it (vision slot loaded or the GUI's "Record depth")
# Installed once per machine with:  (crontab -l 2>/dev/null; echo "@reboot $PWD/vision_service.sh <host>") | crontab -
# Log: ~/rl_deploy/logs/vision_service_<host>.log (truncated at each start).
D=$(dirname "$(readlink -f "$0")")
case "$1" in
  rhps1v) CMD=("python3" "-u" "$D/camera_on_demand.py") ;;
  rhps1c) CMD=("$D/vision_rhps1c.sh" "--camera-host" "${RLQP_CAMERA_HOST:-10.16.236.200:5264}") ;;
  *) echo "usage: $0 rhps1v|rhps1c"; exit 2 ;;
esac
LOG=$HOME/rl_deploy/logs/vision_service_$1.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
sleep 20  # network and drivers up after boot
while true; do
  echo "=== $(date '+%F %T') start ${CMD[*]}" >> "$LOG"
  "${CMD[@]}" >> "$LOG" 2>&1
  echo "=== $(date '+%F %T') exited $?, restart in 5 s" >> "$LOG"
  sleep 5
  [ "$(stat -c %s "$LOG")" -gt 50000000 ] && : > "$LOG"
done
