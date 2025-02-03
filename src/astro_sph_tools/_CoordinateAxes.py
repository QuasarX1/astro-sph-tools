from enum import Enum

class CoordinateAxes(Enum):
    """
    Enumeration giving the three axes on a 3D cartesian grid.
    """
    X = 0
    Y = 1
    Z = 2

    def __str__(self) -> str:
        match self:
            case CoordinateAxes.X:
                return "x"
            case CoordinateAxes.Y:
                return "y"
            case CoordinateAxes.Z:
                return "z"
            case _:
                raise ValueError()

    @staticmethod
    def from_string(value: str) -> "CoordinateAxes":
        match value.strip().lower():
            case "x":
                return CoordinateAxes.X
            case "y":
                return CoordinateAxes.Y
            case "z":
                return CoordinateAxes.Z
            case _:
                raise ValueError()
