from .msg import Mrow
from sensor_msgs.msg import Image
from typing import Any, List

GetCameraImageRequest = Any
class GetCameraImageResponse:
    image: Image
    m_rows: List[Mrow]

    def __init__(self, image: Image = ..., m_rows: List[Mrow] = ...) -> None: ...
GetCameraImage = Any
GetFrameSizeRequest = Any
class GetFrameSizeResponse:
    width: float
    height: float
    def __init__(self, width: float = ..., height: float = ...) -> None: ...
GetFrameSize = Any
GetPoseRequest = Any
GetPoseResponse = Any
GetPose = Any
GetSonarRequest = Any
GetSonarResponse = Any
GetSonar = Any
GetTurtlesRequest = Any
GetTurtlesResponse = Any
GetTurtles = Any
HasTurtleRequest = Any
HasTurtleResponse = Any
HasTurtle = Any
KillRequest = Any
KillResponse = Any
Kill = Any
class SetPenRequest:
    r: int
    g: int
    b: int
    width: int
    off: int
    def __init__(
        self,
        r: int = ...,
        g: int = ...,
        b: int = ...,
        width: int = ...,
        off: int = ...,
    ) -> None: ...
SetPenResponse = Any
SetPen = Any
SpawnRequest = Any
SpawnResponse = Any
Spawn = Any
TeleportAbsoluteRequest = Any
TeleportAbsoluteResponse = Any
TeleportAbsolute = Any
TeleportRelativeRequest = Any
TeleportRelativeResponse = Any
TeleportRelative = Any
