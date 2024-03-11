from std_msgs.msg import Header
from typing import Any
BatteryState = Any
CameraInfo = Any
ChannelFloat32 = Any
CompressedImage = Any
FluidPressure = Any
Illuminance = Any
class Image:
    header: Header
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes
Imu = Any
JointState = Any
Joy = Any
JoyFeedback = Any
JoyFeedbackArray = Any
LaserEcho = Any
LaserScan = Any
MagneticField = Any
MultiDOFJointState = Any
MultiEchoLaserScan = Any
NavSatFix = Any
NavSatStatus = Any
PointCloud = Any
PointCloud2 = Any
PointField = Any
Range = Any
RegionOfInterest = Any
RelativeHumidity = Any
Temperature = Any
TimeReference = Any
