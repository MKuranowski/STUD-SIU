# pyright: basic

import threading
from time import sleep
from typing import Any, List, Optional

import rospy
import turtlesim.msg
import turtlesim.srv
from typing_extensions import Self

from .simulator import CameraCell, Color, ColorChecker, Position, Simulator

WAIT_AFTER_MOVE: Optional[float] = 0.005

_ros_initialization_lock = threading.Lock()
_ros_initialized = False


def _initialize_ros() -> None:
    global _ros_initialization_lock, _ros_initialized
    with _ros_initialization_lock:
        if not _ros_initialized:
            rospy.init_node("siu_example")
            _ros_initialized = True


class ROSSimulator(Simulator):
    def __init__(self) -> None:
        _initialize_ros()
        self.has_turtle_service = rospy.ServiceProxy("has_turtle", turtlesim.srv.HasTurtle)
        self.has_turtle_service.wait_for_service()
        self.kill_turtle_service = rospy.ServiceProxy("kill", turtlesim.srv.Kill)
        self.kill_turtle_service.wait_for_service()
        self.spawn_service = rospy.ServiceProxy("spawn", turtlesim.srv.Spawn)
        self.spawn_service.wait_for_service()
        self.get_pose_service = rospy.ServiceProxy("get_pose", turtlesim.srv.GetPose)
        self.get_pose_service.wait_for_service()
        self.get_camera_service = rospy.ServiceProxy(
            "get_camera_image",
            turtlesim.srv.GetCameraImage,
        )
        self.get_camera_service.wait_for_service()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def has_turtle(self, name: str) -> bool:
        return self.has_turtle_service(name).result

    def kill_turtle(self, name: str) -> None:
        self.kill_turtle_service(name)

    def spawn_turtle(self, name: str, at: Position = Position()) -> None:
        self.spawn_service(at.x, at.y, at.angle, name)
        rospy.ServiceProxy(f"/{name}/set_pen", turtlesim.srv.SetPen)(off=1)
        if WAIT_AFTER_MOVE:
            sleep(WAIT_AFTER_MOVE)

    def get_position(self, name: str) -> Position:
        pose = self.get_pose_service(name).pose
        return Position(pose.x, pose.y, pose.theta)

    def move_absolute(self, name: str, new_pos: Position) -> None:
        rospy.ServiceProxy(
            f"/{name}/teleport_absolute",
            turtlesim.srv.TeleportAbsolute,
        )(x=new_pos.x, y=new_pos.y, theta=new_pos.angle)
        if WAIT_AFTER_MOVE:
            sleep(WAIT_AFTER_MOVE)

    def move_relative(self, name: str, distance: float, angle: float) -> None:
        rospy.ServiceProxy(
            f"/{name}/teleport_absolute",
            turtlesim.srv.TeleportRelative,
        )(linear=distance, angular=angle)
        if WAIT_AFTER_MOVE:
            sleep(WAIT_AFTER_MOVE)

    def read_camera(
        self,
        name: str,
        frame_pixel_size: int,
        cell_side_count: int,
        goal: Position,
    ) -> List[List[CameraCell]]:
        img = self.get_camera_service(
            name=name,
            frame_pixel_size=frame_pixel_size,
            cell_count=cell_side_count * cell_side_count,
            x_offset=0,
            goal=turtlesim.msg.Pose(x=goal.x, y=goal.y),
            show_matrix_cells_and_goal=True,
        )
        return [
            [
                # XXX: cell.occupy is set when **NOT** occupied - no need to negate,
                #      but holy *** is the API awful
                CameraCell(cell.red, cell.green, cell.blue, cell.distance, bool(cell.occupy))
                for cell in row.cells
            ]
            for row in img.m_rows
        ]

    def get_color_checker(self, name: str) -> ColorChecker:
        return ROSColorChecker(name)


class ROSColorChecker(ColorChecker):
    def __init__(self, name: str) -> None:
        self.color: Color = Color(0, 0, 0)
        self.subscriber = rospy.Subscriber(
            f"/{name}/color_sensor",
            turtlesim.msg.Color,
            self.ros_callback,
        )

    def ros_callback(self, data: turtlesim.msg.Color) -> None:
        self.color = Color(
            clamp((data.r - 200) / 50, -1.0, 1.0),
            clamp(data.g / 255, 0.0, 1.0),
            clamp((data.b - 200) / 50, -1.0, 1.0),
        )

    def check(self) -> Color:
        return self.color


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
