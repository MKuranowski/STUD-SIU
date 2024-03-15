FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04
ARG WORKSPACE_FOLDER=/workspace
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -qy
RUN apt-get install -qy --no-install-recommends build-essential curl python3 python3-dev

# http://wiki.ros.org/noetic/Installation/Ubuntu

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update
RUN apt-get install -qy --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    ros-noetic-desktop
RUN ["sh", "-c", "set -ex; . /opt/ros/noetic/setup.sh; rosdep init; rosdep update"]

# The following is from prof's documentation, except that unnecessary packages were removed

RUN apt-get install -qy --no-install-recommends \
    qtbase5-dev \
    python3-tk \
    python3-catkin-tools \
    python3-pip

RUN pip install tensorflow

WORKDIR /opt/siu_ws/src
RUN git clone --branch noetic-devel --depth 1 https://github.com/RCPRG-ROS-pkg/ros_tutorials
WORKDIR /opt/siu_ws
RUN ["sh", "-c", "set -ex; . /opt/ros/noetic/setup.sh; catkin build"]
RUN ln -fs "$WORKSPACE_FOLDER/roads.png" /opt/siu_ws/src/ros_tutorials/turtlesim/images/roads.png
