from math import cos, sin
from typing import Dict, List, Tuple, TypeVar

import cv2

from src.simulator.simulator import Color

from .simulator import CameraCell, ColorChecker, Position, Simulator

Number = TypeVar("Number", int, float)


class SimpleSimulator(Simulator):
    def __init__(self, background_path: str = "roads.png", scale_m_to_px: float = 22.0) -> None:
        self.scale_m_to_px: float = scale_m_to_px
        self.background_path = background_path

        self.background = cv2.imread(self.background_path)
        """A (width, height, 3)-shaped numpy matrix of 0-255 uint8 representing
        the blue, green and red channel values"""

        self.turtles: Dict[str, Position] = {}

    def position_to_pixels(self, p: Position) -> Tuple[int, int]:
        return (
            clamp(int(p.x * self.scale_m_to_px), 0, self.background.shape[0]),
            clamp(
                self.background.shape[1] - int(p.y * self.scale_m_to_px),
                0,
                self.background.shape[1],
            ),
        )

    def has_turtle(self, name: str) -> bool:
        return name in self.turtles

    def kill_turtle(self, name: str) -> None:
        del self.turtles[name]

    def spawn_turtle(self, name: str, at: Position = Position()) -> None:
        if name in self.turtles:
            raise KeyError(name)
        self.turtles[name] = at

    def set_position(self, name: str, new_pos: Position) -> None:
        if name not in self.turtles:
            raise KeyError(name)
        self.turtles[name] = new_pos

    def move(self, name: str, distance: float, angle: float) -> None:
        current_pos = self.turtles[name]
        new_angle = current_pos.angle + angle
        new_x = distance * cos(new_angle)
        new_y = distance * sin(new_angle)
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
        raise NotImplementedError

    def get_color_checker(self, name: str) -> ColorChecker:
        return SimpleColorChecker(self, name)


class SimpleColorChecker(ColorChecker):
    def __init__(self, simple_simulator: SimpleSimulator, name: str) -> None:
        self.simple_simulator = simple_simulator
        self.name = name

    def check(self) -> Color:
        position = self.simple_simulator.get_position(self.name)
        x, y = self.simple_simulator.position_to_pixels(position)
        b, g, r = self.simple_simulator.background[x, y]
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
