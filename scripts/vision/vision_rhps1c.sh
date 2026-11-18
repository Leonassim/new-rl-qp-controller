#!/bin/bash
# On rhps1c: height map from the Jetson's depth (depth_to_rhps1c.sh, UDP 5263) -> controller (obs_format 12, slot 23),
# also published for RViz (rhps1_vision/height_map, /points; rhps1t: rviz2 -d ~/rl_deploy/rhps1_vision.rviz with
# ROS_DOMAIN_ID=200). Kept off the real-time CPU 3. Extra arguments go to rhps1_vision.py (e.g. --record f.npz).
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=200
unset ROS_LOCALHOST_ONLY
# Real mount (2026-11-18, T1): Azure support of the lab URDF + factory depth offset, refitted on the floor at head
# 0 / 12.6 / 25.2 deg with one pose: optical centre 4.0 cm ahead and 0.6 cm above that chain, 4.3 deg down instead of
# 6. Floor flat to 0.3 cm at all three head angles, 24.5 cm box read 24.7-25.6.
CAM="--cam-xyz 0.1190,0.0017,0.1825 --cam-rpy 1.49513,-0.00503,-1.57095"
exec taskset -c 4-7 nice -n 5 python3 -u "$(dirname "$(readlink -f "$0")")/rhps1_vision.py" --source udp --ros $CAM "$@"
