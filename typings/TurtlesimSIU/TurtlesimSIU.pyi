from geometry_msgs.msg import Twist
from turtlesim.msg import Color, Pose
from turtlesim.srv import GetCameraImageResponse, GetFrameSizeResponse, SetPenRequest
from typing import Literal, List, Optional, Sequence, TypedDict

class _Collision(TypedDict):
    name1: str
    name2: str
class ColorSensor:
    owner: str
    colour: Optional[Color]
    def __init__(self, owner: str) -> None: ...
    def topic_callback(self, data: Color) -> None: ...
    def check(self) -> Optional[Color]: ...
class TurtlesimSIU:
    def __init__(self) -> None: ...
    def getFrameSize(self) -> GetFrameSizeResponse: ...
    def getPose(self, turtle_name: str) -> Pose: ...
    def setVel(self, turtle_name: str, vel: Twist) -> bool: ...
    def setPen(self, turtle_name: str, req: SetPenRequest) -> None: ...
    def hasTurtle(self, turtle_name: str) -> bool: ...
    def killTurtle(self, turtle_name: str) -> None: ...
    def spawnTurtle(self, turtle_name: str, pose: Pose) -> None: ...
    def readSonar(
        self,
        fov_center: float,
        fov_range: float,
        range_min: float,
        range_max: float,
        owner: str,
    ) -> float: ...
    def readCamera(
        self,
        name: str = ...,
        frame_pixel_size: int = ...,
        cell_count: int = ...,
        x_offset: int = ...,
        goal: Pose = ...,
        show_matrix_cells_and_goal: bool = ...,
    ) -> GetCameraImageResponse: ...
    def getColisions(self, names: Sequence[str], collision_range: float) -> List[_Collision]: ...
    def setPose(self, turtle_name: str, pose: Pose, mode: Literal["absolute", "relative"] = ...) -> bool: ...
    def pixelsToScale(self) -> float: ...
