from typing import Any

Accel = Any
AccelStamped = Any
AccelWithCovariance = Any
AccelWithCovarianceStamped = Any
Inertia = Any
InertiaStamped = Any
Point = Any
Point32 = Any
PointStamped = Any
Polygon = Any
PolygonStamped = Any
Pose = Any
Pose2D = Any
PoseArray = Any
PoseStamped = Any
PoseWithCovariance = Any
PoseWithCovarianceStamped = Any
Quaternion = Any
QuaternionStamped = Any
Transform = Any
TransformStamped = Any
class Twist:
    linear: "Vector3"
    angular: "Vector3"
    def __init__(self, linear: "Vector3" = ..., angular: "Vector3" = ...) -> None: ...
TwistStamped = Any
TwistWithCovariance = Any
TwistWithCovarianceStamped = Any
class Vector3:
    x: float
    y: float
    z: float
    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...
Vector3Stamped = Any
Wrench = Any
WrenchStamped = Any
