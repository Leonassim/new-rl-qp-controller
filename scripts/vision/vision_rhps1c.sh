#!/bin/bash
# On rhps1c: height map from the Jetson's depth (depth_to_rhps1c.sh, UDP 5263) -> controller (obs_format 12, slot 23),
# also published for RViz (rhps1_vision/height_map, /points; rhps1t: rviz2 -d ~/rl_deploy/rhps1_vision.rviz with
# ROS_DOMAIN_ID=200). Kept off the real-time CPU 3. Extra arguments go to rhps1_vision.py (e.g. --record f.npz).
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=200
unset ROS_LOCALHOST_ONLY
exec taskset -c 4-7 nice -n 5 python3 -u "$(dirname "$(readlink -f "$0")")/rhps1_vision.py" --source udp --ros "$@"
