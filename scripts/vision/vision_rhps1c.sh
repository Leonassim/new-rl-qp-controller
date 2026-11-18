#!/bin/bash
# On rhps1c: height map from the Jetson's depth (depth_to_rhps1c.sh, UDP 5263) -> controller (obs_format 12, slot 23),
# also published for RViz (rhps1_vision/height_map, /points; rhps1t: rviz2 -d ~/rl_deploy/rhps1_vision.rviz with
# ROS_DOMAIN_ID=200). Kept off the real-time CPU 3. Extra arguments go to rhps1_vision.py (e.g. --record f.npz).
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=200
unset ROS_LOCALHOST_ONLY
# Real mount, fitted on the floor with the robot standing (2026-11-18, calib1): 2.0 deg less pitch and 2.3 cm higher
# than the URDF's azure_depth_frame; the floor then reads flat to 0.3 cm from 0.6 to 2.6 m.
CAM="--cam-xyz 0.0710,0.0017,0.1789 --cam-rpy 1.50105,-0.00041,-1.57081"
exec taskset -c 4-7 nice -n 5 python3 -u "$(dirname "$(readlink -f "$0")")/rhps1_vision.py" --source udp --ros $CAM "$@"
