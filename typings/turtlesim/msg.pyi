from typing import List

class Cell:
    red: float
    green: float
    blue: float
    distance: float
    occupy: float
    def __init__(
        self,
        red: float = ...,
        green: float = ...,
        blue: float = ...,
        distance: float = ...,
        occupy: float = ...,
    ) -> None: ...
class Color:
    r: int
    g: int
    b: int
    def __init__(self, r: int = ..., g: int = ..., b: int = ...) -> None: ...
class Mrow:
    cells: List[Cell]
    def __init__(self, cells: List[Cell] = ...) -> None: ...
class Pose:
    x: float
    y: float
    theta: float
    linear_velocity: float
    angular_velocity: float
    def __init__(
        self,
        x: float = ...,
        y: float = ...,
        theta: float = ...,
        linear_velocity: float = ...,
        angular_velocity: float = ...,
    ) -> None: ...
