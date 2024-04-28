#!/usr/bin/env bash
set -ex
export ROS_HOSTNAME=localhost
. /opt/ros/noetic/setup.bash
if [ -f "/opt/siu_ws/devel/setup.bash" ]; then
    . /opt/siu_ws/devel/setup.bash
else
    . /root/siu_ws/devel/setup.bash
fi
roslaunch -v turtlesim siu.launch &
ROS_PID="$!"
trap 'kill $ROS_PID; wait $ROS_PID' EXIT
python3 "$@"
