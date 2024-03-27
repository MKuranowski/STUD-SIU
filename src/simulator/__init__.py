from typing import Callable
from os import getenv

from .simulator import CameraCell, Color, ColorChecker, Simulator, Position

__all__ = ["CameraCell", "Color", "ColorChecker", "create_simulator", "Simulator", "Position"]

forced_backend = getenv("SIU_BACKEND", "").casefold()
has_ros = getenv("ROS_DISTRO", "").casefold() == "noetic"

create_simulator: Callable[[], Simulator]

if forced_backend == "pygame":
    raise NotImplementedError("TODO: pygame backend")
elif forced_backend == "simple":
    from .simple import SimpleSimulator

    create_simulator = SimpleSimulator
elif forced_backend == "ros":
    from .ros import ROSSimulator

    create_simulator = ROSSimulator
elif forced_backend:
    raise ValueError(f"unknown SIU_BACKEND: {forced_backend!r}")
elif has_ros:
    from .ros import ROSSimulator

    create_simulator = ROSSimulator
else:
    from .simple import SimpleSimulator

    create_simulator = SimpleSimulator
