"""Turn per-frame tracker output into clean world-space object tracks."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from clayview.core.camera import Camera
from clayview.core.heading import EgoPath, HeadingConfig, HeadingCues, solve_heading
from clayview.core.projection import ProjectionConfig, footprint_half_extent, project_box
from clayview.core.smoother import FilterConfig, Measurement, State, Step, smooth_sequence
from clayview.core.types import (
    FLAG_HELD,
    FLAG_LOW_CONF,
    FLAG_TRUNCATED,
    Box,
    Detection,
    EgoStep,
    Observation,
)

# Image model output per detection, keyed by (frame, tracker id).
Views = Mapping[tuple[int, int], Sequence[float]]

VEHICLES = frozenset({"car", "truck", "bus"})
RIDERS = frozenset({"motorcycle", "bicycle"})


@dataclass(frozen=True)
class Segment:
    """One continuous shot. Frames are [start_frame, end_frame)."""

    index: int
    start_frame: int
    end_frame: int
    camera: Camera


@dataclass(frozen=True)
class TrackConfig:
    min_duration_s: float = 0.4
    min_detections: int = 6
    hold_s: float = 0.4  # coast this long after the last detection
    exit_hold_s: float = 0.9  # longer if it left through a frame edge: it is passing ego
    max_gap_s: float = 1.5  # a longer gap inside a track is a different object
    vehicle_min_speed: float = 1.8  # m/s; slower than this, motion direction is noise
    person_min_speed: float = 0.7
    ego_speed_doubt: float = 0.4  # ground speed under this share of ego speed may be ego error
    stitch_gap_s: float = 1.0  # rejoin a broken track across a gap this long
    heading_tau_s: float = 0.35
    confidence_alpha: float = 0.25
    hold_decay_tau_s: float = 0.7
    low_confidence: float = 0.3


@dataclass(frozen=True)
class ObjectSample:
    frame: int
    x: float
    y: float
    heading: float
    confidence: float
    flags: tuple[str, ...]
    box: Box | None


@dataclass(frozen=True)
class WorldTrack:
    id: int
    class_name: str
    segment: int
    samples: tuple[ObjectSample, ...]


def build_tracks(
    detections: Sequence[Detection],
    times: Sequence[float],
    segments: Sequence[Segment],
    ego_steps: Sequence[EgoStep],
    track_config: TrackConfig | None = None,
    projection_config: ProjectionConfig | None = None,
    filter_config: FilterConfig | None = None,
    heading_config: HeadingConfig | None = None,
    views: Views | None = None,
) -> list[WorldTrack]:
    """Project, smooth and filter. Output is ordered by first appearance, with clean ids."""
    cfg = track_config or TrackConfig()
    pcfg = projection_config or ProjectionConfig()
    hcfg = heading_config or HeadingConfig()

    grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for detection in detections:
        segment = _segment_of(segments, detection.frame)
        if segment is not None:
            grouped[(segment.index, detection.track_id)].append(detection)

    fragments: list[tuple[int, list[Detection]]] = []
    for (segment_index, _), track_detections in grouped.items():
        track_detections.sort(key=lambda d: d.frame)
        for fragment in _split_at_gaps(track_detections, times, cfg.max_gap_s):
            fragments.append((segment_index, fragment))

    built: list[tuple[int, str, int, tuple[ObjectSample, ...]]] = []
    for segment_index, segment in enumerate(segments):
        in_segment = [fragment for index, fragment in fragments if index == segment_index]
        path = EgoPath(ego_steps, segment.start_frame, segment.end_frame)
        for fragment in stitch_fragments(in_segment, times, cfg.stitch_gap_s):
            track = _build_one(
                fragment,
                segment,
                path,
                times,
                ego_steps,
                cfg,
                pcfg,
                hcfg,
                filter_config,
                views or {},
            )
            if track is not None:
                built.append(track)

    built.sort(key=lambda item: item[0])
    return [
        WorldTrack(id=index + 1, class_name=name, segment=segment, samples=samples)
        for index, (_, name, segment, samples) in enumerate(built)
    ]


def _segment_of(segments: Sequence[Segment], frame: int) -> Segment | None:
    for segment in segments:
        if segment.start_frame <= frame < segment.end_frame:
            return segment
    return None


def _split_at_gaps(
    detections: list[Detection], times: Sequence[float], max_gap_s: float
) -> list[list[Detection]]:
    fragments: list[list[Detection]] = [[detections[0]]]
    for previous, current in itertools.pairwise(detections):
        if times[current.frame] - times[previous.frame] > max_gap_s:
            fragments.append([])
        fragments[-1].append(current)
    return fragments


def stitch_fragments(
    fragments: Sequence[list[Detection]], times: Sequence[float], max_gap_s: float
) -> list[list[Detection]]:
    """Rejoin tracks the detector broke in two, by extrapolating the box across the gap.

    Works in image space, before projection: a broken track's two halves line up far better as boxes
    than as noisy far-range world positions.
    """
    ordered = sorted(fragments, key=lambda fragment: fragment[0].frame)
    merged: list[list[Detection]] = []
    for fragment in ordered:
        match: list[Detection] | None = None
        best_cost = float("inf")
        for candidate in merged:
            cost = _join_cost(candidate, fragment, times, max_gap_s)
            if cost is not None and cost < best_cost:
                match, best_cost = candidate, cost
        if match is None:
            merged.append(list(fragment))
        else:
            match.extend(fragment)
    return merged


def _join_cost(
    earlier: list[Detection], later: list[Detection], times: Sequence[float], max_gap_s: float
) -> float | None:
    """How far the later box starts from where the earlier one was heading. None if no match."""
    tail, head = earlier[-1], later[0]
    gap_s = float(times[head.frame] - times[tail.frame])
    if not 0.0 < gap_s <= max_gap_s:
        return None
    if _class_family(tail.class_name) != _class_family(head.class_name):
        return None

    size_ratio = head.box.width / max(tail.box.width, 1.0)
    if not 0.6 <= size_ratio <= 1.6:
        return None

    reference = earlier[max(len(earlier) - 8, 0)]
    span_s = max(float(times[tail.frame] - times[reference.frame]), 1e-3)
    rate_x = ((tail.box.x1 + tail.box.x2) - (reference.box.x1 + reference.box.x2)) / (2.0 * span_s)
    rate_y = (tail.box.y2 - reference.box.y2) / span_s
    predicted_x = (tail.box.x1 + tail.box.x2) / 2.0 + rate_x * gap_s
    predicted_y = tail.box.y2 + rate_y * gap_s
    distance = math.hypot(
        (head.box.x1 + head.box.x2) / 2.0 - predicted_x, head.box.y2 - predicted_y
    )
    return distance if distance <= 0.8 * max(tail.box.width, head.box.width) else None


def _class_family(class_name: str) -> str:
    """The detector swaps car, truck and bus on one object, and rider classes likewise."""
    if class_name in VEHICLES:
        return "vehicle"
    return "rider" if class_name in RIDERS else class_name


def _vote_class(detections: Sequence[Detection]) -> str:
    """ByteTrack ids are stable but the label flickers between car and truck; let them vote."""
    scores: dict[str, float] = defaultdict(float)
    for detection in detections:
        scores[detection.class_name] += detection.confidence
    return max(scores, key=lambda name: scores[name])


def _left_through_edge(box: Box, camera: Camera, margin: float = 0.04) -> bool:
    edge = margin * camera.image_width
    return (
        box.x1 <= edge
        or box.x2 >= camera.image_width - edge
        or box.y2 >= camera.bottom - margin * camera.image_height
    )


def _build_one(
    detections: list[Detection],
    segment: Segment,
    path: EgoPath,
    times: Sequence[float],
    ego_steps: Sequence[EgoStep],
    cfg: TrackConfig,
    pcfg: ProjectionConfig,
    hcfg: HeadingConfig,
    filter_config: FilterConfig | None,
    views: Views,
) -> tuple[int, str, int, tuple[ObjectSample, ...]] | None:
    camera = segment.camera
    class_name = _vote_class(detections)

    observed: dict[int, tuple[Detection, Observation]] = {}
    for detection in detections:
        observation = project_box(detection.box, class_name, camera, pcfg)
        if observation is not None:
            observed[detection.frame] = (detection, observation)
    if len(observed) < cfg.min_detections:
        return None

    observed_frames = sorted(observed)
    first_frame, last_frame = observed_frames[0], observed_frames[-1]
    if times[last_frame] - times[first_frame] < cfg.min_duration_s:
        return None

    last_box = observed[last_frame][0].box
    hold_s = cfg.exit_hold_s if _left_through_edge(last_box, camera) else cfg.hold_s
    end_frame = last_frame
    while end_frame + 1 < segment.end_frame and times[end_frame + 1] - times[last_frame] <= hold_s:
        end_frame += 1

    first_observation = observed[first_frame][1]
    steps = [
        Step(
            dt=max(float(times[frame + 1] - times[frame]), 1e-3),
            ego=ego_steps[frame] if frame < len(ego_steps) else EgoStep(),
            measurement=_measurement(observed[frame + 1][1]) if frame + 1 in observed else None,
        )
        for frame in range(first_frame, end_frame)
    ]
    states = smooth_sequence(_measurement(first_observation), steps, filter_config)

    frames = list(range(first_frame, end_frame + 1))
    whole_boxes = {
        frame: hit[0].box for frame, hit in observed.items() if FLAG_TRUNCATED not in hit[1].flags
    }
    frame_views = {
        frame: views[(frame, observed[frame][0].track_id)]
        for frame in whole_boxes
        if (frame, observed[frame][0].track_id) in views
    }
    headings = _headings(
        states, steps, class_name, times, frames, whole_boxes, frame_views, path, cfg, hcfg
    )

    samples: list[ObjectSample] = []
    confidence = observed[first_frame][0].confidence
    last_seen = first_frame
    for index, frame in enumerate(frames):
        state = states[index]
        heading = headings[index]
        hit = observed.get(frame)
        flags: set[str] = set()
        box: Box | None = None
        if hit is not None:
            detection, observation = hit
            confidence += cfg.confidence_alpha * (detection.confidence - confidence)
            flags |= observation.flags
            box = detection.box
            last_seen = frame
            held_factor = 1.0
        else:
            flags.add(FLAG_HELD)
            held_s = float(times[frame] - times[last_seen])
            held_factor = math.exp(-held_s / cfg.hold_decay_tau_s)

        final_confidence = confidence * held_factor * _depth_quality(state)
        if final_confidence < cfg.low_confidence:
            flags.add(FLAG_LOW_CONF)

        center_x, center_y = _object_center(state, heading, class_name)
        samples.append(
            ObjectSample(
                frame=frame,
                x=center_x,
                y=center_y,
                heading=heading,
                confidence=final_confidence,
                flags=tuple(sorted(flags)),
                box=box,
            )
        )
    return first_frame, class_name, segment.index, tuple(samples)


def _measurement(observation: Observation) -> Measurement:
    return Measurement(observation.x, observation.y, observation.sigma_x, observation.sigma_y)


def _depth_quality(state: State) -> float:
    """1 when the position is well known for its range, falling towards 0 as it stops being."""
    range_m = max(math.hypot(state.x, state.y), 1.0)
    ratio = state.sigma_position / (0.25 * range_m)
    return 1.0 / (1.0 + ratio * ratio)


def _object_center(state: State, heading: float, class_name: str) -> tuple[float, float]:
    """Move from the near face of the object to its centre, along the viewing ray."""
    bearing = math.atan2(state.x, state.y)
    extent = footprint_half_extent(class_name, heading - bearing)
    return state.x + extent * math.sin(bearing), state.y + extent * math.cos(bearing)


def _headings(
    states: Sequence[State],
    steps: Sequence[Step],
    class_name: str,
    times: Sequence[float],
    frames: Sequence[int],
    boxes: Mapping[int, Box],
    views: Mapping[int, Sequence[float]],
    path: EgoPath,
    cfg: TrackConfig,
    hcfg: HeadingConfig,
) -> list[float]:
    """Heading over the ground in radians: 0 is ego's direction, positive turns right.

    Below walking pace, or when the velocity could be noise or ego-speed error, motion says nothing
    about heading, and box shape and the road decide instead. A parked track gets one heading from
    everything seen of it, because a car seen from a changing angle fixes its own direction.
    """
    min_speed = cfg.person_min_speed if class_name == "person" else cfg.vehicle_min_speed

    # In the ego frame a resting object's heading turns against ego's yaw. Work in a yaw-free frame,
    # so that holding a heading is just keeping a number.
    cumulative_yaw = [0.0]
    for step in steps:
        cumulative_yaw.append(cumulative_yaw[-1] + step.ego.yaw_rad)

    # Ego's own motion leaks into a track's velocity. Travel leaks in directly, and turning sweeps
    # distant objects sideways, so a parked car far off looks fast while ego turns.
    ego_speeds = [
        (abs(step.ego.forward_m) + abs(step.ego.yaw_rad) * math.hypot(state.x, state.y)) / step.dt
        for step, state in zip(steps, states, strict=False)
    ] or [0.0]
    mean_ego_speed = sum(ego_speeds) / len(ego_speeds)
    fixed_vx = [_rotate_velocity(s, cumulative_yaw[i])[0] for i, s in enumerate(states)]
    fixed_vy = [_rotate_velocity(s, cumulative_yaw[i])[1] for i, s in enumerate(states)]
    mean_speed = math.hypot(sum(fixed_vx) / len(states), sum(fixed_vy) / len(states))

    # A parked car's ground speed is whatever error ego speed has, and that error always points back
    # at ego, so it reads as oncoming traffic. Decide per track: a whole pass of a parked car shows
    # this far better than any one frame, and it can't flip partway through.
    parked = mean_speed < max(min_speed, cfg.ego_speed_doubt * mean_ego_speed)

    first_road = path.yaw_at(frames[0])
    cues: list[HeadingCues] = []
    for index, state in enumerate(states):
        speed = math.hypot(state.vx, state.vy)
        ego_speed = ego_speeds[min(index, len(ego_speeds) - 1)]
        needed = max(min_speed, 1.5 * state.sigma_speed, cfg.ego_speed_doubt * ego_speed)
        moving = not parked and speed >= needed
        road, road_offset = path.road_at(frames[index], state.x, state.y)
        cues.append(
            HeadingCues(
                yaw=cumulative_yaw[index],
                bearing=math.atan2(state.x, state.y),
                box=boxes.get(frames[index]),
                motion=math.atan2(fixed_vx[index], fixed_vy[index]) if moving else None,
                motion_certainty=speed / max(state.sigma_speed, 0.05),
                road=road - first_road if class_name != "person" else None,
                road_offset_m=road_offset,
                view=views.get(frames[index]) if class_name == "car" else None,
            )
        )

    dt = [float(times[frames[i + 1]] - times[frames[i]]) for i in range(len(frames) - 1)]
    solved = solve_heading(cues, dt, class_name, not parked, hcfg)

    sines = [math.sin(value) for value in solved]
    cosines = [math.cos(value) for value in solved]
    sines = _two_way_low_pass(sines, dt, cfg.heading_tau_s)
    cosines = _two_way_low_pass(cosines, dt, cfg.heading_tau_s)
    return [
        _wrap(math.atan2(sine, cosine) - cumulative_yaw[index])
        for index, (sine, cosine) in enumerate(zip(sines, cosines, strict=True))
    ]


def _rotate_velocity(state: State, yaw: float) -> tuple[float, float]:
    """Velocity in the frame ego had at the start of the track."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    return cos * state.vx + sin * state.vy, -sin * state.vx + cos * state.vy


def _two_way_low_pass(values: list[float], dt: Sequence[float], tau: float) -> list[float]:
    forward = list(values)
    for index in range(1, len(forward)):
        alpha = 1.0 - math.exp(-dt[index - 1] / tau)
        forward[index] = forward[index - 1] + alpha * (forward[index] - forward[index - 1])
    for index in range(len(forward) - 2, -1, -1):
        alpha = 1.0 - math.exp(-dt[index] / tau)
        forward[index] = forward[index + 1] + alpha * (forward[index] - forward[index + 1])
    return forward


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
