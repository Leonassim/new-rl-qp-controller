#!/bin/bash
# On rhps1v (Jetson): Azure Kinect depth -> rhps1_vision.py --source udp on rhps1c (UDP 5263).
# Starts the ROS driver on a localhost-only domain (depth images never leave the Jetson), then rlqp_depth_send.py;
# Ctrl-C stops both. Driver output in /tmp/rlqp_azure_driver.log.
source /opt/ros/humble/setup.bash
source "$HOME/Azure/azure_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77 DISPLAY=${DISPLAY:-:0}
D=$(dirname "$(readlink -f "$0")")
ros2 launch azure_kinect_ros_driver driver.launch.py depth_mode:=WFOV_2X2BINNED depth_unit:=16UC1 \
  color_enabled:=false fps:=${RLQP_DEPTH_FPS:-15} point_cloud:=false rgb_point_cloud:=false \
  overwrite_robot_description:=false > /tmp/rlqp_azure_driver.log 2>&1 &
DRV=$!
trap 'kill -INT $DRV 2>/dev/null; wait $DRV 2>/dev/null' EXIT
sleep 3
python3 -u "$D/rlqp_depth_send.py" "$@"
