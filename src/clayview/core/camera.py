"""Pinhole road-plane camera model.

Conventions: image x right, image y down. World is the ego frame on the road plane,
x right and y forward, in metres, with the origin under the camera.

Ground projection treats the camera as level with the horizon at `horizon_y`, so a small
pitch is just a shift of the horizon row: d = f * h / (y - horizon). Lens distortion is
ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_HEIGHT_M = 1.3
DEFAULT_VFOV_DEG = 60.0
DEFAULT_HORIZON_FRACTION = 0.5


@dataclass(frozen=True)
class Camera:
    image_width: int
    image_height: int
    height_m: float = DEFAULT_HEIGHT_M
    vfov_deg: float = DEFAULT_VFOV_DEG
    horizon_y: float | None = None
    center_x: float | None = None
    visible_bottom: float | None = None

    @property
    def focal_px(self) -> float:
        return (self.image_height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def cx(self) -> float:
        return self.image_width / 2.0 if self.center_x is None else self.center_x

    @property
    def horizon(self) -> float:
        if self.horizon_y is None:
            return self.image_height * DEFAULT_HORIZON_FRACTION
        return self.horizon_y

    @property
    def bottom(self) -> float:
        """Lowest image row that shows road. Below it is the frame edge, hood or dash."""
        return float(self.image_height) if self.visible_bottom is None else self.visible_bottom

    @property
    def pitch_deg(self) -> float:
        """Camera pitch implied by the horizon row. Positive looks down."""
        return math.degrees(math.atan2(self.image_height / 2.0 - self.horizon, self.focal_px))

    def ground_distance(self, y_px: float) -> float | None:
        """Forward distance of a ground point at image row y_px, or None at or above the horizon."""
        below = y_px - self.horizon
        if below <= 0.0:
            return None
        return self.focal_px * self.height_m / below

    def lateral(self, x_px: float, distance_m: float) -> float:
        """Lateral offset of image column x_px at a forward distance."""
        return (x_px - self.cx) * distance_m / self.focal_px

    def ground_point(self, x_px: float, y_px: float) -> tuple[float, float] | None:
        """Project an image point to the road plane as (x, y) metres."""
        distance = self.ground_distance(y_px)
        if distance is None:
            return None
        return self.lateral(x_px, distance), distance

    def size_distance(self, real_m: float, size_px: float) -> float:
        """Distance implied by a known real size spanning size_px pixels."""
        return self.focal_px * real_m / size_px

    def image_point(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Inverse of ground_point: road-plane metres to image pixels."""
        return (
            self.cx + x_m * self.focal_px / y_m,
            self.horizon + self.focal_px * self.height_m / y_m,
        )


def horizon_from_pitch(image_height: int, vfov_deg: float, pitch_deg: float) -> float:
    """Horizon row for a camera pitched down by pitch_deg (negative looks up)."""
    focal = (image_height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return image_height / 2.0 - focal * math.tan(math.radians(pitch_deg))
