from typing import List, NamedTuple, Protocol


class CameraCell(NamedTuple):
    """CameraCell represents a single cell returned by the camera"""

    r: float = 0.0
    """r is the average value of the red channel, normalized to [-1, 1] (from [150, 250])"""

    g: float = 0.0
    """g is the average value of the green channel, normalized to [0, 1] (from [0, 255])"""

    b: float = 0.0
    """b is the average value of the blue channel, normalized to [-1, 1] (from [150, 250])"""

    distance_to_goal: float = 0.0
    """distance_to_goal represents the distance to the goal (in meters)
    from the center of the cell"""

    free: bool = True
    """free is set to true if there is no other turtle in the cell"""


class Color(NamedTuple):
    r: float
    """r is the value of the red channel, normalized to [-1, 1] (from [150, 250])"""

    g: float
    """g is the average value of the green channel, normalized to [0, 1] (from [0, 255])"""

    b: float
    """b is the average value of the blue channel, normalized to [-1, 1] (from [150, 250])"""


class Position(NamedTuple):
    x: float = 0.0
    """x is the distance from the left edge of the map, in meters"""

    y: float = 0.0
    """y is the distance from the bottom edge of the map, in meters"""

    angle: float = 0.0
    """angle is the counter-clockwise difference from rightwards, in radians"""


class ColorChecker(Protocol):
    def check(self) -> Color: ...


class Simulator(Protocol):
    def has_turtle(self, name: str) -> bool: ...
    def kill_turtle(self, name: str) -> None: ...
    def spawn_turtle(self, name: str, at: Position = ...) -> None: ...
    def move_absolute(self, name: str, new_pos: Position) -> None: ...
    def move_relative(self, name: str, distance: float, angle: float) -> None: ...
    def get_position(self, name: str) -> Position: ...
    def read_camera(
        self,
        name: str,
        frame_pixel_size: int,
        cell_side_count: int,
        goal: Position,
    ) -> List[List[CameraCell]]: ...
    def get_color_checker(self, name: str) -> ColorChecker: ...
