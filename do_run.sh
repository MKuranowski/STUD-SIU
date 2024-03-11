#!/usr/bin/env bash
set -ex
export ROS_HOSTNAME=localhost
. /opt/ros/noetic/setup.bash
. /opt/siu_ws/devel/setup.bash
roslaunch -v turtlesim siu.launch &
ROS_PID="$!"
trap 'kill $ROS_PID; wait $ROS_PID' EXIT
python3 "$@"
