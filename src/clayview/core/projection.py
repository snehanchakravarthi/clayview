"""Turn a detection box into a road-plane observation.

Two independent distance estimates are fused by inverse variance:

- ground: the box bottom sits on the road, d = f*h/(y - horizon). Very accurate up close, but
  one pixel of error is worth d^2/(f*h) metres, so it blows up near the horizon.
- size: a class has a typical real size, d = f*W/width_px. Proportional error at any range, and
  it is the only option when the bottom of the box is cut off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from clayview.core.camera import Camera
from clayview.core.types import (
    FLAG_CLAMPED,
    FLAG_NO_GROUND,
    FLAG_SIZE_PRIOR,
    FLAG_TRUNCATED,
    Box,
    Observation,
)


@dataclass(frozen=True)
class ClassPrior:
    """Typical real dimensions in metres. `use_height` ranks height over width as the prior."""

    width_m: float
    height_m: float
    length_m: float
    use_height: bool = False


CLASS_PRIORS: dict[str, ClassPrior] = {
    "car": ClassPrior(1.8, 1.5, 4.5),
    "truck": ClassPrior(2.5, 3.2, 8.0),
    "bus": ClassPrior(2.55, 3.2, 12.0),
    "motorcycle": ClassPrior(0.8, 1.5, 2.1, use_height=True),
    "bicycle": ClassPrior(0.6, 1.7, 1.8, use_height=True),
    "person": ClassPrior(0.5, 1.7, 0.4, use_height=True),
}


@dataclass(frozen=True)
class ProjectionConfig:
    max_range_m: float = 120.0
    min_range_m: float = 2.0
    edge_margin_px: float = 3.0
    sigma_bottom_px: float = 2.0  # jitter of the box bottom edge
    sigma_bottom_box_fraction: float = 0.02  # plus this much of the box height
    sigma_size_real: float = 0.14  # relative spread of real sizes within a class
    sigma_size_px: float = 2.0  # jitter of the measured size
    sigma_lateral_px: float = 3.0
    min_ground_below_horizon_px: float = 4.0
    oblique_aspect: float = 1.3  # wider than this, a vehicle box shows more than its width


def project_box(
    box: Box,
    class_name: str,
    camera: Camera,
    config: ProjectionConfig | None = None,
) -> Observation | None:
    """Project one box to the road plane, or return None if nothing in it can be trusted."""
    cfg = config or ProjectionConfig()
    prior = CLASS_PRIORS[class_name]
    margin = cfg.edge_margin_px
    cut_left = box.x1 <= margin
    cut_right = box.x2 >= camera.image_width - margin
    cut_top = box.y1 <= margin
    cut_bottom = box.y2 >= camera.bottom - margin

    flags: set[str] = set()
    if cut_left or cut_right or cut_top or cut_bottom:
        flags.add(FLAG_TRUNCATED)

    focal = camera.focal_px
    ground = _ground_estimate(box, camera, cut_bottom, cfg)
    if ground is None:
        flags.add(FLAG_NO_GROUND)
    size = _size_estimate(box, prior, camera, cut_left or cut_right, cut_top or cut_bottom, cfg)

    if ground is None and size is None:
        return None

    distance, sigma, ground_weight = _fuse(ground, size)
    if ground_weight < 0.5:
        flags.add(FLAG_SIZE_PRIOR)

    if distance > cfg.max_range_m:
        distance = cfg.max_range_m
        flags.add(FLAG_CLAMPED)
    distance = max(distance, cfg.min_range_m)

    # A box cut by a side edge keeps one honest edge; rebuild the centre from it.
    center_x = (box.x1 + box.x2) / 2.0
    expected_width_px = focal * prior.width_m / distance
    if cut_left and not cut_right:
        center_x = box.x2 - expected_width_px / 2.0
    elif cut_right and not cut_left:
        center_x = box.x1 + expected_width_px / 2.0

    x = camera.lateral(center_x, distance)
    sigma_x = math.hypot(distance * cfg.sigma_lateral_px / focal, x * sigma / distance)
    return Observation(
        x=x,
        y=distance,
        sigma_x=sigma_x,
        sigma_y=sigma,
        ground_weight=ground_weight,
        flags=frozenset(flags),
    )


def _ground_estimate(
    box: Box, camera: Camera, cut_bottom: bool, cfg: ProjectionConfig
) -> tuple[float, float] | None:
    """(distance, sigma) from the ground contact, or None if there isn't a usable one."""
    if cut_bottom or box.y2 - camera.horizon < cfg.min_ground_below_horizon_px:
        return None
    distance = camera.ground_distance(box.y2)
    if distance is None:
        return None
    sigma_px = cfg.sigma_bottom_px + cfg.sigma_bottom_box_fraction * box.height
    sigma = distance * distance * sigma_px / (camera.focal_px * camera.height_m)
    return distance, sigma


def _size_estimate(
    box: Box,
    prior: ClassPrior,
    camera: Camera,
    cut_horizontal: bool,
    cut_vertical: bool,
    cfg: ProjectionConfig,
) -> tuple[float, float] | None:
    """(distance, sigma) from the class size prior, using the dimension the box can vouch for."""
    if box.width <= 0.0 or box.height <= 0.0:
        return None
    # A close car cut by the hood looks wide only because its height is missing.
    oblique = not cut_vertical and box.width / box.height > cfg.oblique_aspect
    prefer_height = prior.use_height or oblique

    if prefer_height and not cut_vertical:
        real_m, size_px = prior.height_m, box.height
    elif not cut_horizontal and not oblique:
        real_m, size_px = prior.width_m, box.width
    elif not cut_vertical:
        real_m, size_px = prior.height_m, box.height
    else:
        return None

    distance = camera.size_distance(real_m, size_px)
    sigma = distance * math.hypot(cfg.sigma_size_real, cfg.sigma_size_px / size_px)
    return distance, sigma


def _fuse(
    ground: tuple[float, float] | None, size: tuple[float, float] | None
) -> tuple[float, float, float]:
    """Inverse-variance fusion. Returns (distance, sigma, weight given to the ground estimate)."""
    if ground is None:
        assert size is not None
        return size[0], size[1], 0.0
    if size is None:
        return ground[0], ground[1], 1.0
    weight_ground = 1.0 / (ground[1] * ground[1])
    weight_size = 1.0 / (size[1] * size[1])
    total = weight_ground + weight_size
    distance = (weight_ground * ground[0] + weight_size * size[0]) / total
    return distance, math.sqrt(1.0 / total), weight_ground / total


def footprint_half_extent(class_name: str, relative_angle_rad: float) -> float:
    """Half the footprint along a viewing ray, for an object at that angle to the ray.

    The box bottom is the near face of the object. Pushing the point out by this much along the
    ray puts it at the object's centre, which is where the viewer places the model.
    """
    prior = CLASS_PRIORS[class_name]
    return (
        abs(prior.length_m * math.cos(relative_angle_rad))
        + abs(prior.width_m * math.sin(relative_angle_rad))
    ) / 2.0
