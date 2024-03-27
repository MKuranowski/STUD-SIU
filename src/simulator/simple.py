from typing import List

from .simulator import CameraCell, ColorChecker, Position, Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        pass

    def has_turtle(self, name: str) -> bool:
        raise NotImplementedError

    def kill_turtle(self, name: str) -> None:
        raise NotImplementedError

    def spawn_turtle(self, name: str, at: Position = ...) -> None:
        raise NotImplementedError

    def set_position(self, name: str, new_pos: Position) -> None:
        raise NotImplementedError

    def move(self, name: str, distance: float, angle: float) -> None:
        raise NotImplementedError

    def get_position(self, name: str) -> Position:
        raise NotImplementedError

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
