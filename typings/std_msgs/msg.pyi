from typing import Any, Protocol

class _genpy_Time(Protocol):
    # NOTE: This should be genpy.Time, but I can't be bothered to typehint that
    secs: int
    nsecs: int

Bool = Any
Byte = Any
ByteMultiArray = Any
Char = Any
ColorRGBA = Any
Duration = Any
Empty = Any
Float32 = Any
Float32MultiArray = Any
Float64 = Any
Float64MultiArray = Any
class Header:
    seq: int
    time: _genpy_Time
    frame_id: str
    # NOTE: This class also has a proper __init__ function
Int16 = Any
Int16MultiArray = Any
Int32 = Any
Int32MultiArray = Any
Int64 = Any
Int64MultiArray = Any
Int8 = Any
Int8MultiArray = Any
MultiArrayDimension = Any
MultiArrayLayout = Any
String = Any
Time = Any
UInt16 = Any
UInt16MultiArray = Any
UInt32 = Any
UInt32MultiArray = Any
UInt64 = Any
UInt64MultiArray = Any
UInt8 = Any
UInt8MultiArray = Any
