"""Build the static street from labelled frames: road edges, paint, trees, poles and buildings.

Everything here works in one shot's world frame, which is ego's pose at the start of the shot:
x right, y forward. Ego's own path links the frames, so a curb seen at 20 m lines up with the same
curb seen at 10 m. The road surface lives on a strip that follows that path, one row per station
along it and one column per metre-fraction across it, which keeps a long drive small and turns
"where are the road edges" into scanning each row outwards from the middle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from clayview.core.camera import Camera
from clayview.core.heading import EgoPath
from clayview.core.tracks import Segment
from clayview.core.types import EgoStep

# Mapillary Vistas class ids, as the segmentation model numbers them.
CURB, CURB_CUT = 2, 9
FENCE, GUARD_RAIL, BARRIER, WALL = 3, 4, 5, 6
BIKE_LANE, CROSSWALK_PLAIN, PARKING, PEDESTRIAN_AREA, ROAD, SERVICE_LANE, SIDEWALK = (
    7, 8, 10, 11, 13, 14, 15,
)  # fmt: skip
BUILDING = 17
CROSSWALK_PAINT, LANE_PAINT = 23, 24
SAND, TERRAIN, VEGETATION = 26, 29, 30
CATCH_BASIN, MANHOLE, POTHOLE = 36, 41, 43
STREET_LIGHT, POLE, SIGN_FRAME, UTILITY_POLE, TRAFFIC_LIGHT, SIGN_BACK, SIGN_FRONT = (
    44, 45, 46, 47, 48, 49, 50,
)  # fmt: skip

# What a ground cell can be. Cars, people and the ego hood say nothing about the ground beneath.
G_ROAD, G_LANE_PAINT, G_CROSSWALK, G_CURB, G_WALK, G_GRASS, G_OTHER = range(7)
GROUPS = 7
NO_GROUP = 255
PAINT_UPSCALE = 3

UPRIGHT = (STREET_LIGHT, POLE, SIGN_FRAME, UTILITY_POLE, TRAFFIC_LIGHT, SIGN_BACK, SIGN_FRONT)
BARRIERS = {FENCE: "fence", GUARD_RAIL: "guardRail", BARRIER: "barrier", WALL: "wall"}


def _group_table() -> np.ndarray:
    table = np.full(256, NO_GROUP, dtype=np.uint8)
    road = (BIKE_LANE, CROSSWALK_PLAIN, PARKING, ROAD, SERVICE_LANE, CATCH_BASIN, MANHOLE, POTHOLE)
    table[list(road)] = G_ROAD
    table[LANE_PAINT] = G_LANE_PAINT
    table[CROSSWALK_PAINT] = G_CROSSWALK
    table[[CURB, CURB_CUT]] = G_CURB
    table[[SIDEWALK, PEDESTRIAN_AREA]] = G_WALK
    table[[TERRAIN, SAND]] = G_GRASS
    table[[BUILDING, VEGETATION, FENCE, GUARD_RAIL, BARRIER, WALL]] = G_OTHER
    return table


GROUP_OF = _group_table()


@dataclass(frozen=True)
class SegMaps:
    """Label maps. `maps[i]` belongs to processed frame `frames[i]`."""

    frames: np.ndarray  # (n,) int32
    maps: np.ndarray  # (n, height, width) uint8


@dataclass(frozen=True)
class WorldConfig:
    row_m: float = 0.4  # strip size along the path
    col_m: float = 0.2  # and across it
    half_width_m: float = 20.0
    near_m: float = 3.0
    far_m: float = 28.0
    max_slope: float = 0.9  # past this, lens distortion bends the ground too much to trust
    paint_share: float = 0.22  # of a cell's votes, to count as paint
    paint_min_votes: int = 4
    paint_min_area_m2: float = 0.14
    edge_gap_m: float = 0.8  # this much non-road ends the road
    edge_smooth_m: float = 8.0
    min_road_m: float = 1.2
    default_half_road_m: float = 3.5
    max_road_m: float = 16.0
    station_m: float = 2.0
    upright_range_m: float = 40.0
    upright_merge_m: float = 2.0
    upright_min_views: int = 3
    column_step_px: int = 6
    tree_range_m: float = 32.0
    tree_min_points: int = 8
    tree_min_height_m: float = 3.5
    wall_range_m: float = 35.0
    wall_bin_m: float = 6.0
    wall_min_points: int = 4


@dataclass(frozen=True)
class Path:
    """Ego's path resampled to even steps, so a strip row is a fixed distance."""

    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    frame_s: np.ndarray  # path position of each frame of the shot

    def right(self) -> tuple[np.ndarray, np.ndarray]:
        return np.cos(self.yaw), -np.sin(self.yaw)

    def at(self, s: float) -> tuple[float, float, float]:
        return (
            float(np.interp(s, self.s, self.x)),
            float(np.interp(s, self.s, self.y)),
            float(np.interp(s, self.s, self.yaw)),
        )


def resample_path(ego: EgoPath, step_m: float) -> Path:
    travelled = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(ego.x), np.diff(ego.y)))))
    stations = np.arange(0.0, max(travelled[-1], step_m) + step_m, step_m)
    moving = np.concatenate(([True], np.diff(travelled) > 1e-6))
    knots = travelled[moving]
    if len(knots) < 2:
        knots = np.array([0.0, step_m])
        xs, ys, yaws = np.zeros(2), np.array([0.0, step_m]), np.zeros(2)
    else:
        xs, ys, yaws = ego.x[moving], ego.y[moving], ego.yaw[moving]
    return Path(
        s=stations,
        x=np.interp(stations, knots, xs),
        y=np.interp(stations, knots, ys),
        yaw=np.interp(stations, knots, yaws),
        frame_s=travelled,
    )


def to_ego(
    world_x: np.ndarray, world_y: np.ndarray, ego_x: float, ego_y: float, yaw: float
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = world_x - ego_x, world_y - ego_y
    cos, sin = math.cos(yaw), math.sin(yaw)
    return dx * cos - dy * sin, dx * sin + dy * cos


def to_world(x: np.ndarray, y: np.ndarray, ego_x: float, ego_y: float, yaw: float) -> np.ndarray:
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.stack((ego_x + x * cos + y * sin, ego_y - x * sin + y * cos), axis=-1)


def collect_ground(
    path: Path,
    ego: EgoPath,
    camera: Camera,
    frames: Sequence[int],
    maps: np.ndarray,
    cfg: WorldConfig,
) -> np.ndarray:
    """Vote for what each strip cell is, in every labelled frame that can see it."""
    columns = round(2.0 * cfg.half_width_m / cfg.col_m)
    offsets = (np.arange(columns) + 0.5) * cfg.col_m - cfg.half_width_m
    votes = np.zeros((len(path.s), columns, GROUPS), dtype=np.uint16)
    right_x, right_y = path.right()
    height, width = maps.shape[1:]
    scale_x, scale_y = width / camera.image_width, height / camera.image_height

    for map_index, frame in enumerate(frames):
        local = frame - ego.start_frame
        position = float(path.frame_s[local])
        first = max(int((position + cfg.near_m) / cfg.row_m), 0)
        last = min(int((position + cfg.far_m) / cfg.row_m) + 1, len(path.s))
        if last <= first:
            continue
        rows = np.arange(first, last)
        world_x = path.x[rows, None] + right_x[rows, None] * offsets
        world_y = path.y[rows, None] + right_y[rows, None] * offsets
        # The camera looks along the car's heading plus its own turn on the mount.
        looking = float(ego.yaw[local] + ego.mount[local])
        forward_x, forward_y = to_ego(
            world_x, world_y, float(ego.x[local]), float(ego.y[local]), looking
        )
        usable = (forward_y > cfg.near_m * 0.7) & (
            np.abs(forward_x) < cfg.max_slope * np.maximum(forward_y, 0.1)
        )
        safe_y = np.where(usable, forward_y, 1.0)
        u = (camera.cx + camera.focal_px * forward_x / safe_y) * scale_x
        v = (camera.horizon + camera.focal_px * camera.height_m / safe_y) * scale_y
        usable &= (u >= 0) & (u < width) & (v >= 0) & (v < camera.bottom * scale_y)
        row_index, column_index = np.nonzero(usable)
        groups = GROUP_OF[maps[map_index][v[usable].astype(int), u[usable].astype(int)]]
        known = groups != NO_GROUP
        # Near cells are sharper, so they count for far more. A thin painted line only wins a cell
        # from close up, and a small pitch error moves distant rows by metres.
        ahead = forward_y[usable]
        weight = np.select([ahead < 9.0, ahead < 15.0], [6, 2], default=1).astype(np.uint16)
        votes[first + row_index[known], column_index[known], groups[known]] += weight[known]
    return votes


def _median_fill(values: np.ndarray, window: int) -> np.ndarray:
    """Median over a sliding window, ignoring NaN. Short gaps fill, long ones stay NaN."""
    half = window // 2
    out = np.full_like(values, np.nan)
    for index in range(len(values)):
        chunk = values[max(index - half, 0) : index + half + 1]
        known = chunk[np.isfinite(chunk)]
        if len(known) * 3 >= len(chunk):
            out[index] = np.median(known)
    return out


def _low_pass(values: np.ndarray, window: int) -> np.ndarray:
    known = np.isfinite(values)
    if not known.any():
        return values
    filled = np.interp(np.arange(len(values)), np.nonzero(known)[0], values[known])
    kernel = np.ones(window) / window
    smooth = np.convolve(np.pad(filled, window // 2, mode="edge"), kernel, mode="same")
    smooth = smooth[window // 2 : window // 2 + len(values)]
    return np.where(known, smooth, np.nan)


def _fill_edge(edge: np.ndarray, fallback: float) -> np.ndarray:
    """Carry an edge across rows that couldn't see it. A row with no data at all gets a lane."""
    known = np.isfinite(edge)
    if not known.any():
        return np.full_like(edge, fallback)
    return np.interp(np.arange(len(edge)), np.nonzero(known)[0], edge[known])


def _scan_edge(is_road: np.ndarray, known: np.ndarray, middle: int, step: int, gap: int) -> int:
    """Walk outwards from the middle of a row. Returns the last road cell, or -1 if none."""
    edge = -1
    misses = 0
    column = middle
    while 0 <= column < len(is_road):
        if is_road[column]:
            edge, misses = column, 0
        elif known[column]:
            misses += 1
            if misses >= gap:
                break
        column += step
    return edge


def road_edges(votes: np.ndarray, cfg: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
    """Left and right road edges per strip row, in metres from the path. Left is negative."""
    surface = (
        votes[..., G_ROAD].astype(np.int32) + votes[..., G_LANE_PAINT] + votes[..., G_CROSSWALK]
    )
    elsewhere = votes[..., [G_CURB, G_WALK, G_GRASS, G_OTHER]].sum(axis=2).astype(np.int32)
    is_road = surface > elsewhere
    known = surface + elsewhere > 0
    columns = votes.shape[1]
    middle = columns // 2
    gap = max(round(cfg.edge_gap_m / cfg.col_m), 1)
    left = np.full(len(votes), np.nan)
    right = np.full(len(votes), np.nan)
    span = max(round(1.0 / cfg.col_m), 1)
    for row in range(len(votes)):
        if not is_road[row, middle - span : middle + span].any():
            continue
        low = _scan_edge(is_road[row], known[row], middle, -1, gap)
        high = _scan_edge(is_road[row], known[row], middle, 1, gap)
        if low >= 0:
            left[row] = (low - columns / 2.0) * cfg.col_m
        if high >= 0:
            right[row] = (high + 1 - columns / 2.0) * cfg.col_m
    left[(left > -cfg.min_road_m) | (left < -cfg.max_road_m)] = np.nan
    right[(right < cfg.min_road_m) | (right > cfg.max_road_m)] = np.nan
    window = max(round(cfg.edge_smooth_m / cfg.row_m), 3) | 1
    return (
        _low_pass(_median_fill(left, window), window // 2 | 1),
        _low_pass(_median_fill(right, window), window // 2 | 1),
    )


def _side_of_road(
    votes: np.ndarray, edge: np.ndarray, outward: int, cfg: WorldConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What lies beyond an edge, per row: curb seen, sidewalk width, and share of grass."""
    columns = votes.shape[1]
    reach = round(5.0 / cfg.col_m)
    curb = np.zeros(len(votes), dtype=bool)
    walk = np.zeros(len(votes))
    grass = np.zeros(len(votes))
    near = max(round(1.0 / cfg.col_m), 1)
    for row in range(len(votes)):
        if not np.isfinite(edge[row]):
            continue
        start = round(edge[row] / cfg.col_m + columns / 2.0)
        cells = [start + outward * step for step in range(-near, reach)]
        cells = [cell for cell in cells if 0 <= cell < columns]
        if not cells:
            continue
        here = votes[row, cells]
        curb[row] = here[: 2 * near, G_CURB].sum() > 0.1 * max(here[: 2 * near].sum(), 1)
        beyond = here[near:]
        winners = beyond.argmax(axis=1)[beyond.sum(axis=1) > 0]
        walk[row] = (winners == G_WALK).sum() * cfg.col_m
        grass[row] = (winners == G_GRASS).sum() * cfg.col_m
    return curb, walk, grass


def extract_paint(
    votes: np.ndarray, group: int, path: Path, cfg: WorldConfig
) -> list[list[list[float]]]:
    """Painted shapes as polygons in world coordinates."""
    total = votes.sum(axis=2).astype(np.float32)
    mask = (votes[..., group] >= cfg.paint_min_votes) & (
        votes[..., group] >= cfg.paint_share * np.maximum(total, 1.0)
    )
    image = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Contours run through cell centres, so a line one cell wide would collapse to nothing.
    image = cv2.resize(
        image, None, fx=PAINT_UPSCALE, fy=PAINT_UPSCALE, interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    columns = votes.shape[1]
    right_x, right_y = path.right()
    cell_area = cfg.row_m * cfg.col_m / PAINT_UPSCALE**2
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) * cell_area < cfg.paint_min_area_m2:
            continue
        simple = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2) / PAINT_UPSCALE
        if len(simple) < 3:
            continue
        rows = np.clip(np.round(simple[:, 1]).astype(int), 0, len(path.s) - 1)
        offsets = (simple[:, 0] - columns / 2.0) * cfg.col_m
        xs = path.x[rows] + right_x[rows] * offsets
        ys = path.y[rows] + right_y[rows] * offsets
        polygons.append(
            [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys, strict=True)]
        )
    return polygons


def _contact_row(mask: np.ndarray, column: int, ground: np.ndarray, first_row: int) -> int | None:
    """The lowest row where this column's mask stands on the ground."""
    rows = np.nonzero(mask[first_row:, column])[0]
    if not len(rows):
        return None
    for row in rows[::-1] + first_row:
        below = min(row + 3, len(mask) - 1)
        if ground[below, column]:
            return int(row)
    return None


def _ground_mask(label: np.ndarray) -> np.ndarray:
    groups = GROUP_OF[label]
    return np.isin(groups, (G_ROAD, G_LANE_PAINT, G_CROSSWALK, G_CURB, G_WALK, G_GRASS))


def uprights_in_frame(
    label: np.ndarray, camera: Camera, cfg: WorldConfig
) -> list[tuple[str, float, float, float]]:
    """Poles and what hangs on them, as (kind, x, y, height) in ego's frame."""
    mask = np.isin(label, UPRIGHT).astype(np.uint8)
    count, ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ground = _ground_mask(label)
    height, width = label.shape
    scale_x, scale_y = camera.image_width / width, camera.image_height / height
    found: list[tuple[str, float, float, float]] = []
    for component in range(1, count):
        left, top, box_width, box_height, area = stats[component]
        if area < 12 or box_height < 8:
            continue
        bottom = top + box_height - 1
        ground_row = min(bottom + 3, height - 1)
        columns = np.nonzero(ids[bottom - min(4, box_height - 1)] == component)[0]
        if not len(columns):
            columns = np.array([left + box_width // 2])
        column = int(np.median(columns))
        if not ground[ground_row, column]:
            continue  # the foot is hidden, so the row says nothing about distance
        point = camera.ground_point(column * scale_x, (bottom + 1) * scale_y)
        if point is None or not 2.0 < point[1] < cfg.upright_range_m:
            continue
        below = (bottom + 1) * scale_y - camera.horizon
        rise = box_height * scale_y / below * camera.height_m
        pixels = label[ids == component]
        kind = "pole"
        if (pixels == TRAFFIC_LIGHT).sum() >= 6:
            kind = "trafficLight"
        elif np.isin(pixels, (SIGN_FRONT, SIGN_BACK)).sum() >= 10:
            kind = "sign"
        elif (pixels == STREET_LIGHT).sum() >= 6:
            kind = "streetLight"
        found.append((kind, point[0], point[1], rise))
    return found


def merge_uprights(
    seen: Sequence[tuple[str, float, float, float]], cfg: WorldConfig
) -> list[dict[str, Any]]:
    """Join repeat sightings of one object. Anything seen too few times was a mistake."""
    clusters: list[dict[str, Any]] = []
    for kind, x, y, height in seen:
        best, best_distance = None, cfg.upright_merge_m
        for cluster in clusters:
            distance = math.hypot(x - cluster["x"], y - cluster["y"])
            if distance < best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append({"x": x, "y": y, "items": [(kind, x, y, height)]})
        else:
            best["items"].append((kind, x, y, height))
            best["x"] = float(np.median([item[1] for item in best["items"]]))
            best["y"] = float(np.median([item[2] for item in best["items"]]))

    order = ("trafficLight", "sign", "streetLight", "pole")
    objects: list[dict[str, Any]] = []
    for cluster in clusters:
        items = cluster["items"]
        if len(items) < cfg.upright_min_views:
            continue
        kinds = [item[0] for item in items]
        kind = next((name for name in order if kinds.count(name) * 3 >= len(kinds)), "pole")
        objects.append(
            {
                "kind": kind,
                "x": round(cluster["x"], 2),
                "y": round(cluster["y"], 2),
                "height": round(float(np.percentile([i[3] for i in items], 75)), 2),
            }
        )
    return objects


def column_contacts(
    mask: np.ndarray, ground: np.ndarray, camera: Camera, cfg: WorldConfig, reach_m: float
) -> list[tuple[float, float, float]]:
    """Where a mask meets the ground, column by column, as (x, y, height) in ego's frame."""
    height, width = mask.shape
    scale_x, scale_y = camera.image_width / width, camera.image_height / height
    first_row = int(camera.horizon / scale_y) + 2
    found: list[tuple[float, float, float]] = []
    for column in range(0, width, cfg.column_step_px):
        row = _contact_row(mask, column, ground, first_row)
        if row is None:
            continue
        point = camera.ground_point(column * scale_x, (row + 1) * scale_y)
        if point is None or not 2.0 < point[1] < reach_m:
            continue
        top = int(np.argmax(mask[:, column]))
        below = (row + 1) * scale_y - camera.horizon
        found.append((point[0], point[1], (row - top) * scale_y / below * camera.height_m))
    return found


def pick_trees(points: np.ndarray, cfg: WorldConfig) -> list[dict[str, Any]]:
    """Turn vegetation contact points into separate plants, tallest evidence first."""
    if not len(points):
        return []
    cells = np.round(points[:, :2] / 1.5).astype(np.int64)
    keys, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    heights = np.zeros(len(keys))
    centers = np.zeros((len(keys), 2))
    for index in range(len(keys)):
        members = points[inverse == index]
        heights[index] = np.percentile(members[:, 2], 70)
        centers[index] = members[:, :2].mean(axis=0)

    objects: list[dict[str, Any]] = []
    alive = counts >= cfg.tree_min_points
    for index in np.argsort(-(counts * np.minimum(heights, 20.0))):
        if not alive[index]:
            continue
        height = float(np.clip(heights[index], 1.0, 24.0))
        radius = float(np.clip(0.3 * height, 0.9, 6.0))
        kind = "tree" if height >= cfg.tree_min_height_m else "bush"
        objects.append(
            {
                "kind": kind,
                "x": round(float(centers[index, 0]), 2),
                "y": round(float(centers[index, 1]), 2),
                "height": round(height, 2),
                "radius": round(radius, 2),
            }
        )
        spacing = max(1.3 * radius, 2.0)
        alive &= np.hypot(*(centers - centers[index]).T) > spacing
    return objects


def _to_strip(points: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Nearest path station and signed offset (right is positive) of world points."""
    stations = np.zeros(len(points), dtype=np.int64)
    step = max(len(path.s) // 4000, 1)
    for index, point in enumerate(points):
        distances = np.hypot(path.x[::step] - point[0], path.y[::step] - point[1])
        stations[index] = int(np.argmin(distances)) * step
    right_x, right_y = path.right()
    offsets = (points[:, 0] - path.x[stations]) * right_x[stations] + (
        points[:, 1] - path.y[stations]
    ) * right_y[stations]
    return stations, offsets


def wall_runs(points: np.ndarray, path: Path, cfg: WorldConfig) -> list[dict[str, Any]]:
    """Group contact points into straight-ish runs along each side of the path."""
    if not len(points):
        return []
    stations, offsets = _to_strip(points, path)
    positions = path.s[stations]
    bins = (positions / cfg.wall_bin_m).astype(np.int64)
    right_x, right_y = path.right()
    runs: list[dict[str, Any]] = []
    for side, sign in (("left", -1.0), ("right", 1.0)):
        chosen = offsets * sign > 0.5
        current: list[list[float]] = []
        heights: list[float] = []
        last_bin: int | None = None
        for bin_index in np.unique(bins[chosen]):
            members = chosen & (bins == bin_index)
            if members.sum() < cfg.wall_min_points:
                continue
            if last_bin is not None and bin_index - last_bin > 2:
                runs.extend(_finish_run(side, current, heights))
                current, heights = [], []
            offset = float(np.median(offsets[members]))
            row = min(int((bin_index + 0.5) * cfg.wall_bin_m / path.s[1]), len(path.s) - 1)
            current.append(
                [
                    float(path.x[row] + right_x[row] * offset),
                    float(path.y[row] + right_y[row] * offset),
                ]
            )
            heights.append(float(np.percentile(points[members, 2], 70)))
            last_bin = int(bin_index)
        runs.extend(_finish_run(side, current, heights))
    return runs


def _finish_run(side: str, points: list[list[float]], heights: list[float]) -> list[dict[str, Any]]:
    if len(points) < 2:
        return []
    return [
        {
            "side": side,
            "height": round(float(np.percentile(heights, 70)), 2),
            "points": [[round(x, 2), round(y, 2)] for x, y in points],
        }
    ]


def build_segment(
    segment: Segment,
    ego_steps: Sequence[EgoStep],
    seg: SegMaps,
    cfg: WorldConfig,
) -> dict[str, Any]:
    """The static world of one shot."""
    ego = EgoPath(ego_steps, segment.start_frame, segment.end_frame)
    path = resample_path(ego, cfg.row_m)
    chosen = (seg.frames >= segment.start_frame) & (seg.frames < segment.end_frame)
    frames = [int(frame) for frame in seg.frames[chosen]]
    maps = seg.maps[chosen]
    camera = segment.camera

    votes = collect_ground(path, ego, camera, frames, maps, cfg)
    left, right = road_edges(votes, cfg)
    measured = np.isfinite(left) & np.isfinite(right)
    left, right = (
        _fill_edge(left, -cfg.default_half_road_m),
        _fill_edge(right, cfg.default_half_road_m),
    )
    left_curb, left_walk, left_grass = _side_of_road(votes, left, -1, cfg)
    right_curb, right_walk, right_grass = _side_of_road(votes, right, 1, cfg)

    uprights: list[tuple[str, float, float, float]] = []
    trees: list[tuple[float, float, float]] = []
    walls: dict[str, list[tuple[float, float, float]]] = {name: [] for name in BARRIERS.values()}
    buildings: list[tuple[float, float, float]] = []

    def place(
        points: Sequence[tuple[float, float, float]], pose: tuple[float, float, float]
    ) -> Any:
        array = np.array(points)
        world = to_world(array[:, 0], array[:, 1], *pose)
        return [
            (float(a), float(b), float(c)) for (a, b), c in zip(world, array[:, 2], strict=True)
        ]

    for label, frame in zip(maps, frames, strict=True):
        local = frame - segment.start_frame
        pose = (float(ego.x[local]), float(ego.y[local]), float(ego.yaw[local] + ego.mount[local]))
        ground = _ground_mask(label)
        for kind, x, y, height in uprights_in_frame(label, camera, cfg):
            point = to_world(np.array(x), np.array(y), *pose)
            uprights.append((kind, float(point[0]), float(point[1]), height))
        contacts = column_contacts(label == VEGETATION, ground, camera, cfg, cfg.tree_range_m)
        if contacts:
            trees.extend(place(contacts, pose))
        contacts = column_contacts(label == BUILDING, ground, camera, cfg, cfg.wall_range_m)
        if contacts:
            buildings.extend(place(contacts, pose))
        for class_id, name in BARRIERS.items():
            contacts = column_contacts(label == class_id, ground, camera, cfg, cfg.wall_range_m)
            if contacts:
                walls[name].extend(place(contacts, pose))

    step = max(round(cfg.station_m / cfg.row_m), 1)
    stations: list[dict[str, Any]] = []
    for row in range(0, len(path.s), step):
        rows = slice(max(row - step, 0), row + step + 1)
        stations.append(
            {
                "s": round(float(path.s[row]), 2),
                "x": round(float(path.x[row]), 2),
                "y": round(float(path.y[row]), 2),
                "yaw": round(float(path.yaw[row]), 4),
                "left": round(float(left[row]), 2),
                "right": round(float(right[row]), 2),
                "measured": bool(measured[rows].mean() > 0.5),
                "leftCurb": bool(left_curb[rows].mean() > 0.4),
                "rightCurb": bool(right_curb[rows].mean() > 0.4),
                "leftWalk": round(float(np.median(left_walk[rows])), 2),
                "rightWalk": round(float(np.median(right_walk[rows])), 2),
                "leftGrass": round(float(np.median(left_grass[rows])), 2),
                "rightGrass": round(float(np.median(right_grass[rows])), 2),
            }
        )

    plants = pick_trees(np.array(trees).reshape(-1, 3), cfg)
    barriers: list[dict[str, Any]] = []
    for name, points in walls.items():
        for run in wall_runs(np.array(points).reshape(-1, 3), path, cfg):
            barriers.append({"kind": name, **run})
    return {
        "segment": segment.index,
        "road": stations,
        "markings": [
            *[
                {"kind": "line", "points": polygon}
                for polygon in extract_paint(votes, G_LANE_PAINT, path, cfg)
            ],
            *[
                {"kind": "crosswalk", "points": polygon}
                for polygon in extract_paint(votes, G_CROSSWALK, path, cfg)
            ],
        ],
        "objects": [*merge_uprights(uprights, cfg), *plants],
        "buildings": wall_runs(np.array(buildings).reshape(-1, 3), path, cfg),
        "barriers": barriers,
    }


def build_world(
    segments: Sequence[Segment],
    ego_steps: Sequence[EgoStep],
    seg: SegMaps,
    config: WorldConfig | None = None,
) -> dict[str, Any]:
    cfg = config or WorldConfig()
    return {"segments": [build_segment(s, ego_steps, seg, cfg) for s in segments]}
