from math import cos, sin
from typing import Dict, List

from .simulator import CameraCell, ColorChecker, Position, Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        self.turtles: Dict[str, Position] = {}

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
        cell_count: int,
        goal: Position,
    ) -> List[List[CameraCell]]:
        raise NotImplementedError

    def get_color_checker(self, name: str) -> ColorChecker:
        raise NotImplementedError
