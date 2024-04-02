from math import cos, dist, sin
from typing import Any, Dict, List, Tuple, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .simulator import CameraCell, Color, ColorChecker, Position, Simulator

USE_BROKEN_ROS_DISTANCE_ALGORITHM = False


Number = TypeVar("Number", int, float)
NDArrayFloat = npt.NDArray[np.float_]
NDArrayFloat32 = npt.NDArray[np.float32]
NDArrayUint8 = npt.NDArray[np.uint8]


class SimpleSimulator(Simulator):
    def __init__(self, background_path: str = "roads.png", scale_m_to_px: float = 22.0) -> None:
        self.scale: float = scale_m_to_px
        """Scale of the image, in px/m"""

        self.background_path = background_path
        """Path to the background image"""

        self.background: NDArrayUint8 = cv2.imread(self.background_path)  # type: ignore
        """A (height, width, 3)-shaped numpy matrix of 0-255 uint8 representing
        the blue, green and red channel values"""

        self.turtles: Dict[str, Position] = {}
        """Positions of all known turtles, indexed by their name"""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def position_to_pixels(self, p: Position) -> Tuple[int, int]:
        """position_to_pixels converts a given position in meters, to
        corresponding x and y indices on the background image"""
        return (
            clamp(int(p.x * self.scale), 0, self.background.shape[1] - 1),
            clamp(
                self.background.shape[0] - int(p.y * self.scale),
                0,
                self.background.shape[0] - 1,
            ),
        )

    def pixels_to_position(self, x: float, y: float) -> Tuple[float, float]:
        """pixels_to_position converts x and y background image indices
        to x and y offsets in meters"""
        return x / self.scale, (self.background.shape[0] - y) / self.scale

    def has_turtle(self, name: str) -> bool:
        return name in self.turtles

    def kill_turtle(self, name: str) -> None:
        del self.turtles[name]

    def spawn_turtle(self, name: str, at: Position = Position()) -> None:
        if name in self.turtles:
            raise KeyError(name)
        self.turtles[name] = at

    def move_absolute(self, name: str, new_pos: Position) -> None:
        if name not in self.turtles:
            raise KeyError(name)
        self.turtles[name] = new_pos

    def move_relative(self, name: str, distance: float, angle: float) -> None:
        current_pos = self.turtles[name]
        new_angle = current_pos.angle + angle
        new_x = current_pos.x + distance * cos(new_angle)
        new_y = current_pos.y + distance * sin(new_angle)
        self.turtles[name] = Position(new_x, new_y, new_angle)

    def get_position(self, name: str) -> Position:
        return self.turtles[name]

    def read_camera(
        self,
        name: str,
        frame_pixel_size: int,
        cell_side_count: int,
        goal: Position,
    ) -> List[List[CameraCell]]:
        bg_to_camera: NDArrayFloat32 = self._bg_to_camera_transform_matrix(
            self.turtles[name],
            frame_pixel_size,
        )
        camera_to_bg: NDArrayFloat = cv2.invertAffineTransform(bg_to_camera)  # type: ignore
        camera: NDArrayUint8 = cv2.warpAffine(  # type: ignore
            self.background,
            bg_to_camera,
            (frame_pixel_size, frame_pixel_size),
        )

        cell_pixel_size = int(frame_pixel_size / cell_side_count)

        return [
            [
                self._read_cell(
                    camera=camera,
                    row_idx=j,
                    col_idx=cell_side_count - i - 1,
                    cell_pixel_size=cell_pixel_size,
                    camera_to_bg=camera_to_bg,
                    goal=goal,
                )
                for j in range(cell_side_count)
            ]
            for i in range(cell_side_count)
        ]

    def _read_cell(
        self,
        camera: NDArrayUint8,
        row_idx: int,
        col_idx: int,
        cell_pixel_size: int,
        camera_to_bg: NDArrayFloat32,
        goal: Position,
    ) -> CameraCell:
        """_read_cell returns the CameraCell corresponding to a 2D-slice of a camera view"""
        mean_r, mean_g, mean_b = self._read_cell_colors(camera, row_idx, col_idx, cell_pixel_size)

        dist_algorithm = (
            self._calc_cell_distance
            if not USE_BROKEN_ROS_DISTANCE_ALGORITHM
            else self._calc_cell_distance_with_broken_ros_algorithm
        )
        dist = dist_algorithm(row_idx, col_idx, cell_pixel_size, camera_to_bg, goal)

        if len(self.turtles) > 1:
            raise NotImplementedError("TODO: Calculate CameraCell.free")

        return CameraCell(
            r=clamp((mean_r - 200) / 50, -1.0, 1.0),
            g=clamp(mean_g / 255, -1.0, 1.0),
            b=clamp((mean_b - 200) / 50, -1.0, 1.0),
            distance_to_goal=dist,
            free=True,
        )

    def _read_cell_colors(
        self,
        camera: NDArrayUint8,
        row_idx: int,
        col_idx: int,
        cell_pixel_size: int,
    ) -> Tuple[float, float, float]:
        """_read_cell_colors returns the mean value r, g and b of the 2D-slice of camera,
        as defined by the row and colum indices"""
        cell = camera[
            row_idx * cell_pixel_size : (row_idx + 1) * cell_pixel_size,
            col_idx * cell_pixel_size : (col_idx + 1) * cell_pixel_size,
            :,
        ]
        mean_b, mean_g, mean_r = np.mean(cell, axis=(0, 1))
        return mean_r, mean_g, mean_b

    def _calc_cell_distance(
        self,
        row_idx: int,
        col_idx: int,
        cell_pixel_size: int,
        camera_to_bg: NDArrayFloat32,
        goal: Position,
    ) -> float:
        """_calc_cell_distance returns the distance from the cell to the goal.
        The cell is identified by its row and column indices, and the camera_to_bg
        matrix is used to move from camera to global coordinate system."""
        cell_center_cam_x_px = (col_idx + 0.5) * cell_pixel_size
        cell_center_cam_y_px = (row_idx + 0.5) * cell_pixel_size
        cell_center_bg_x_px, cell_center_bg_y_px = (
            camera_to_bg @ np.array([cell_center_cam_x_px, cell_center_cam_y_px, 1.0]).T
        )
        cell_center_bg_x, cell_center_bg_y = self.pixels_to_position(
            cell_center_bg_x_px,
            cell_center_bg_y_px,
        )
        return dist((goal.x, goal.y), (cell_center_bg_x, cell_center_bg_y))

    def _calc_cell_distance_with_broken_ros_algorithm(
        self,
        row_idx: int,
        col_idx: int,
        cell_pixel_size: int,
        camera_to_bg: NDArrayFloat32,
        goal: Position,
    ) -> float:
        """_calc_cell_distance returns the "distance" from the cell to the goal,
        introducing errors as they are present in the original, prof's code.
        The cell is identified by its row and column indices, and the camera_to_bg
        matrix is used to move from camera to global coordinate system."""
        cell_center_cam_x_px = (col_idx + 0.5) * cell_pixel_size
        cell_center_cam_y_px = (row_idx + 0.5) * cell_pixel_size
        cell_center_bg_x_px, cell_center_bg_y_px = (
            camera_to_bg @ np.array([cell_center_cam_x_px, cell_center_cam_y_px, 1.0]).T
        )

        # Introduce error due to wrong height calculation from
        # https://github.com/RCPRG-ros-pkg/ros_tutorials/blob/noetic-devel/turtlesim/src/turtle_frame.cpp#L153
        height_m = (self.background.shape[0] - 1) / self.scale
        # Which also propagates to cell_center_bg_y_px through:
        # https://github.com/RCPRG-ros-pkg/ros_tutorials/blob/noetic-devel/turtlesim/src/turtle_frame.cpp#L248
        # and https://github.com/RCPRG-ros-pkg/ros_tutorials/blob/noetic-devel/turtlesim/src/turtle_frame.cpp#L375
        cell_center_bg_y_px -= 1

        # Introduce error due to x-coordinate truncation from
        # https://github.com/RCPRG-ros-pkg/ros_tutorials/blob/noetic-devel/turtlesim/src/turtle_frame.cpp#L419
        goal_x_px = int(goal.x) * self.scale

        # Introduce error due to y-coordinate truncation and +1 from
        # https://github.com/RCPRG-ros-pkg/ros_tutorials/blob/noetic-devel/turtlesim/src/turtle_frame.cpp#L420
        goal_y_px = int(height_m + 1) * self.scale - int(goal.y * self.scale)

        return (
            dist((goal_x_px, goal_y_px), (cell_center_bg_x_px, cell_center_bg_y_px)) / self.scale
        )

    def _bg_to_camera_transform_matrix(
        self,
        position: Position,
        resolution: int,
    ) -> NDArrayFloat32:
        """_bg_to_camera_transform_matrix returns the affine transformation matrix
        for moving from the global/background coordinate system to camera coordinate system.

        The camera coordinate system is a (resolution, resolution) square box,
        with (0,0) representing the top-left corner of the camera view and the turtle
        pointing straight upwards (pi/2), positioned at the very bottom, half-way left-to-right:
        (resolution/2, resolution).

        >>> x_camera, y_camera = bg_to_camera_matrix @ [x_global, y_global, 1]ᵀ

        To obtain a matrix for moving from the camera to global coordinate system,
        use cv2.invertAffineTransform.

        >>> camera_to_bg_matrix = cv2.invertAffineTransform(bg_to_camera_matrix)
        >>> x_global, y_global = camera_to_bg_matrix @ [x_camera, y_camera, 1]ᵀ

        See https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html for details
        on affine transformations in general.
        """
        # A turtle positioned at `o`, headed in the direction indicated by the arrow,
        # has a camera view denoted by the square with corners 1..4 (left diagram).
        # In order to provide the camera view, the background image needs be transformed
        # to contain exactly the bounding box, oriented as indicated by the right diagram.
        # 4---1      1---2
        # |   |      |   |
        # o→C |      | C |
        # |   |      | ↑ |
        # 3---2      4-o-3
        # To perform this transformation, we're using an affine transformation (cv2.wrapAffine).
        # To find the affine transformation matrix, we're using the cv2.getAffineTransform
        # function on the 4-o-C triangle.
        # Finding the coordinates of the 4, o and C points in the camera coordinates is trivial.
        # The coordinates of "o" in the global coordinates are provided, however the coordinates
        # of "4" and "C" need to be calculated. The diagram below describes the necessary math:
        #                |sin α|  cos α  |
        #              --------------------> x
        #             |
        #      -|     |  4
        #       |     |   \
        # cos α |     |    \
        #       |    -|     \         ___C
        #       |     |      \     ___
        #       |sin α|       \ ___
        #      -|    -|        o
        #             |
        #             ↓
        #             y
        resolution_half = resolution * 0.5
        x, y = self.position_to_pixels(position)
        bg_triangle = np.array(
            [
                # "4" - corner coordinates
                [
                    x - resolution_half * sin(position.angle),
                    y - resolution_half * cos(position.angle),
                ],
                # "o" - turtle coordinates
                [x, y],
                # "C" - center coordinates
                [
                    x + resolution_half * cos(position.angle),
                    y - resolution_half * sin(position.angle),
                ],
            ],
            dtype=np.float32,
        )
        camera_trinagle = np.array(
            [
                [0.0, resolution],  # "4" - corner coordinates
                [resolution_half, resolution],  # "o" - turtle coordinates
                [resolution_half, resolution_half],  # "C" - center coordinates
            ],
            dtype=np.float32,
        )
        return cv2.getAffineTransform(bg_triangle, camera_trinagle)  # type: ignore

    def get_color_checker(self, name: str) -> ColorChecker:
        return SimpleColorChecker(self, name)


class SimpleColorChecker(ColorChecker):
    def __init__(self, simple_simulator: SimpleSimulator, name: str) -> None:
        self.simple_simulator = simple_simulator
        self.name = name

    def check(self) -> Color:
        position = self.simple_simulator.get_position(self.name)
        x, y = self.simple_simulator.position_to_pixels(position)
        b, g, r = self.simple_simulator.background[y, x]
        return Color(
            r=clamp((r - 200) / 50, -1.0, 1.0),
            g=clamp(g / 255, 0.0, 1.0),
            b=clamp((b - 200) / 50, -1.0, 1.0),
        )


def clamp(x: Number, lo: Number, hi: Number) -> Number:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
