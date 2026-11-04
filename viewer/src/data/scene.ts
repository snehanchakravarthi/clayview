/** The scene file contract, and fast sampling of it at any time. */

export const SCENE_VERSION = 3;

export const OBJECT_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"] as const;
export type ObjectClass = (typeof OBJECT_CLASSES)[number];

export interface SceneObject {
  id: number;
  class: ObjectClass;
  x: number;
  y: number;
  heading: number;
  confidence: number;
  flags: string[];
  box?: [number, number, number, number];
}

/** Ego in its shot's world frame: x right, y forward at the shot's first frame, yaw right. */
export type ScenePose = [x: number, y: number, yaw: number];

export interface SceneFrame {
  frame: number;
  t: number;
  segment: number;
  /**
   * `yaw` is the car's turn to the next frame. `mount` is how far the camera is turned on its mount
   * relative to the car, positive right: 0 on a dashcam, not on a phone in a hand, a helmet or a head.
   */
  ego: { speed: number; yaw: number; mount: number; pose: ScenePose };
  objects: SceneObject[];
}

export type WorldPoint = [x: number, y: number];

/** A cross-section of the real road, every 2 m along ego's path. */
export interface RoadStation {
  s: number;
  x: number;
  y: number;
  yaw: number;
  /** Road edges in metres along the path's right vector: left is negative, right positive. */
  left: number;
  right: number;
  /** False when the edges were interpolated or defaulted. */
  measured: boolean;
  leftCurb: boolean;
  rightCurb: boolean;
  leftWalk: number;
  rightWalk: number;
  leftGrass: number;
  rightGrass: number;
}

export interface RoadMarking {
  kind: "line" | "crosswalk";
  /** A closed polygon in world metres. Concave and noisy. */
  points: WorldPoint[];
}

export const STREET_OBJECT_KINDS = [
  "tree",
  "bush",
  "pole",
  "streetLight",
  "trafficLight",
  "sign",
] as const;
export type StreetObjectKind = (typeof STREET_OBJECT_KINDS)[number];

export interface StreetObject {
  kind: StreetObjectKind;
  x: number;
  y: number;
  height: number;
  /** On trees and bushes only. */
  radius?: number;
}

export interface BuildingRun {
  side: "left" | "right";
  /** Capped by what the frame showed, so this is a hint. */
  height: number;
  points: WorldPoint[];
}

export interface BarrierRun {
  kind: "wall" | "fence" | "guardRail" | "barrier";
  side: "left" | "right";
  height: number;
  points: WorldPoint[];
}

/** The static street of one shot. Poses restart at each shot, so worlds don't share a frame. */
export interface WorldSegment {
  segment: number;
  road: RoadStation[];
  markings: RoadMarking[];
  objects: StreetObject[];
  buildings: BuildingRun[];
  barriers: BarrierRun[];
}

export interface SceneWorld {
  segments: WorldSegment[];
}

export interface CameraInfo {
  heightM: number;
  heightSource: string;
  vfovDeg: number;
  focalPx: number;
  horizonY: number;
  horizonFraction: number;
  horizonSource: string;
  pitchDeg: number;
  centerX: number;
  visibleBottom: number;
}

export interface SceneSegment {
  index: number;
  startFrame: number;
  endFrame: number;
  start: number;
  end: number;
  camera: CameraInfo;
}

export interface SceneMetadata {
  source: {
    path: string;
    width: number;
    height: number;
    fps: number;
    duration: number;
    frameCount: number;
    processedFrames: number;
  };
  camera: CameraInfo | null;
  model: Record<string, unknown>;
  egoMotion: { mode: string; coverage: number };
}

export interface SceneFile {
  version: number;
  metadata: SceneMetadata;
  segments: SceneSegment[];
  frames: SceneFrame[];
  /** Null when the pipeline ran with --no-world: the viewer falls back to its decorative road. */
  world?: SceneWorld | null;
}

export interface Track {
  id: number;
  className: ObjectClass;
  times: Float64Array;
  x: Float32Array;
  y: Float32Array;
  heading: Float32Array;
  confidence: Float32Array;
  flags: string[][];
  boxes: (SceneObject["box"] | null)[];
  start: number;
  end: number;
}

export interface Timeline {
  scene: SceneFile;
  tracks: Track[];
  times: Float64Array;
  speed: Float32Array;
  distance: Float64Array;
  curvature: Float32Array;
  density: Float32Array;
  poseX: Float64Array;
  poseY: Float64Array;
  /** Unwrapped within a shot, so interpolation never spins the long way round. */
  poseYaw: Float64Array;
  mount: Float32Array;
  duration: number;
  cuts: number[];
}

export interface ObjectSample {
  id: number;
  className: ObjectClass;
  x: number;
  y: number;
  heading: number;
  confidence: number;
  flags: string[];
  box: SceneObject["box"] | null;
  fade: number;
}

export interface EgoPose {
  x: number;
  y: number;
  yaw: number;
}

export interface EgoSample {
  speed: number;
  distance: number;
  curvature: number;
  segment: number;
  pose: EgoPose;
  /** Ego's yaw eased over a moment, for a camera that leans into turns instead of snapping. */
  easedYaw: number;
  /** The video camera's yaw on its mount, relative to the car. It looks along pose.yaw + mount. */
  mount: number;
}

export class SceneError extends Error {}

const FADE_SECONDS = 0.35;
const MAX_CURVATURE = 1 / 70;
const CURVATURE_WINDOW_SECONDS = 1.2;
const MIN_CURVING_SPEED = 2;
const EASED_YAW_SECONDS = 0.9;
const EASED_YAW_SAMPLES = 8;

/** Parse and validate scene text. Throws SceneError with a message fit to show the user. */
export function parseScene(text: string): SceneFile {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new SceneError("That file isn't valid JSON.");
  }
  const scene = value as Partial<SceneFile> | null;
  if (!scene || typeof scene !== "object" || !Array.isArray(scene.frames) || !scene.metadata) {
    throw new SceneError("That JSON isn't a clayview scene: it has no frames or metadata.");
  }
  if (scene.version !== SCENE_VERSION) {
    throw new SceneError(
      `This scene is version ${String(scene.version)}, and the viewer reads version ${SCENE_VERSION}. Re-run clayview process.`,
    );
  }
  if (scene.frames.length === 0) {
    throw new SceneError("This scene has no frames.");
  }
  return scene as SceneFile;
}

/** Index the per-frame file into per-track typed arrays, so sampling is a binary search. */
export function buildTimeline(scene: SceneFile): Timeline {
  const frames = scene.frames;
  const count = frames.length;
  const times = new Float64Array(count);
  const speed = new Float32Array(count);
  const distance = new Float64Array(count);
  const density = new Float32Array(count);
  const yawRate = new Float32Array(count);
  const poseX = new Float64Array(count);
  const poseY = new Float64Array(count);
  const poseYaw = new Float64Array(count);
  const mount = new Float32Array(count);

  const grouped = new Map<number, { t: number; object: SceneObject }[]>();
  for (let index = 0; index < count; index++) {
    const frame = frames[index]!;
    times[index] = frame.t;
    speed[index] = frame.ego.speed;
    density[index] = frame.objects.length;
    const pose = frame.ego.pose;
    poseX[index] = pose[0];
    poseY[index] = pose[1];
    const sameShot = index > 0 && frames[index - 1]!.segment === frame.segment;
    poseYaw[index] = sameShot ? nearestAngle(pose[2], poseYaw[index - 1]!) : pose[2];
    mount[index] = frame.ego.mount;
    if (index > 0) {
      const dt = frame.t - frames[index - 1]!.t;
      distance[index] = distance[index - 1]! + speed[index - 1]! * dt;
      if (dt > 0) yawRate[index - 1] = frames[index - 1]!.ego.yaw / dt;
    }
    for (const object of frame.objects) {
      let entries = grouped.get(object.id);
      if (!entries) {
        entries = [];
        grouped.set(object.id, entries);
      }
      entries.push({ t: frame.t, object });
    }
  }

  const tracks: Track[] = [];
  for (const [id, entries] of grouped) {
    const length = entries.length;
    const track: Track = {
      id,
      className: entries[0]!.object.class,
      times: new Float64Array(length),
      x: new Float32Array(length),
      y: new Float32Array(length),
      heading: new Float32Array(length),
      confidence: new Float32Array(length),
      flags: new Array<string[]>(length),
      boxes: new Array<SceneObject["box"] | null>(length),
      start: entries[0]!.t,
      end: entries[length - 1]!.t,
    };
    let previousHeading = entries[0]!.object.heading;
    for (let index = 0; index < length; index++) {
      const { t, object } = entries[index]!;
      track.times[index] = t;
      track.x[index] = object.x;
      track.y[index] = object.y;
      previousHeading = nearestAngle(object.heading, previousHeading);
      track.heading[index] = previousHeading;
      track.confidence[index] = object.confidence;
      track.flags[index] = object.flags;
      track.boxes[index] = object.box ?? null;
    }
    tracks.push(track);
  }

  return {
    scene,
    tracks,
    times,
    speed,
    distance,
    curvature: curvatureSeries(times, speed, yawRate),
    density,
    poseX,
    poseY,
    poseYaw,
    mount,
    duration: scene.metadata.source.duration || times[count - 1]!,
    cuts: scene.segments.slice(1).map((segment) => segment.start),
  };
}

function curvatureSeries(
  times: Float64Array,
  speed: Float32Array,
  yawRate: Float32Array,
): Float32Array {
  const count = times.length;
  const result = new Float32Array(count);
  let left = 0;
  let right = 0;
  let sumYaw = 0;
  let sumSpeed = 0;
  for (let index = 0; index < count; index++) {
    const t = times[index]!;
    while (right < count && times[right]! <= t + CURVATURE_WINDOW_SECONDS) {
      sumYaw += yawRate[right]!;
      sumSpeed += speed[right]!;
      right++;
    }
    while (left < right && times[left]! < t - CURVATURE_WINDOW_SECONDS) {
      sumYaw -= yawRate[left]!;
      sumSpeed -= speed[left]!;
      left++;
    }
    const samples = Math.max(right - left, 1);
    const meanSpeed = sumSpeed / samples;
    if (meanSpeed < MIN_CURVING_SPEED) continue;
    const curvature = sumYaw / samples / meanSpeed;
    result[index] = Math.max(-MAX_CURVATURE, Math.min(MAX_CURVATURE, curvature));
  }
  return result;
}

/** The angle equal to `angle` modulo a full turn that sits closest to `reference`. */
export function nearestAngle(angle: number, reference: number): number {
  const turn = Math.PI * 2;
  return angle + Math.round((reference - angle) / turn) * turn;
}

/** Largest index with values[index] <= value, clamped to the array. */
export function lowerIndex(values: Float64Array, value: number): number {
  let low = 0;
  let high = values.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (values[middle]! <= value) low = middle;
    else high = middle - 1;
  }
  return low;
}

function ease(value: number): number {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

/** Interpolated state of a track at time t, or null when it isn't alive then. */
export function sampleTrack(track: Track, t: number): ObjectSample | null {
  if (t < track.start || t > track.end) return null;
  const index = lowerIndex(track.times, t);
  const next = Math.min(index + 1, track.times.length - 1);
  const span = track.times[next]! - track.times[index]!;
  const mix = span > 0 ? (t - track.times[index]!) / span : 0;
  const nearest = mix < 0.5 ? index : next;
  return {
    id: track.id,
    className: track.className,
    x: lerp(track.x[index]!, track.x[next]!, mix),
    y: lerp(track.y[index]!, track.y[next]!, mix),
    heading: lerp(track.heading[index]!, track.heading[next]!, mix),
    confidence: lerp(track.confidence[index]!, track.confidence[next]!, mix),
    flags: track.flags[nearest]!,
    box: track.boxes[nearest] ?? null,
    fade: ease((t - track.start) / FADE_SECONDS) * ease((track.end - t) / FADE_SECONDS),
  };
}

export function sampleEgo(timeline: Timeline, t: number): EgoSample {
  const { times, speed, distance, curvature, scene } = timeline;
  const index = lowerIndex(times, t);
  const next = Math.min(index + 1, times.length - 1);
  const span = times[next]! - times[index]!;
  const mix = span > 0 ? Math.max(0, Math.min(1, (t - times[index]!) / span)) : 0;
  const segment = scene.frames[index]!.segment;
  // Poses restart at a cut, so the last frame of a shot holds instead of leaping to the next.
  const poseMix = scene.frames[next]!.segment === segment ? mix : 0;
  const pose = {
    x: lerp(timeline.poseX[index]!, timeline.poseX[next]!, poseMix),
    y: lerp(timeline.poseY[index]!, timeline.poseY[next]!, poseMix),
    yaw: lerp(timeline.poseYaw[index]!, timeline.poseYaw[next]!, poseMix),
  };
  return {
    speed: lerp(speed[index]!, speed[next]!, mix),
    distance: lerp(distance[index]!, distance[next]!, mix),
    curvature: lerp(curvature[index]!, curvature[next]!, mix),
    segment,
    pose,
    easedYaw: easedYaw(timeline, t, segment, pose.yaw),
    mount: lerp(timeline.mount[index]!, timeline.mount[next]!, poseMix),
  };
}

/** Yaw at a moment of this shot, clamped to the shot's own frames. */
function yawInShot(timeline: Timeline, t: number, segment: number): number | null {
  const { times, poseYaw, scene } = timeline;
  const index = lowerIndex(times, t);
  if (scene.frames[index]!.segment !== segment) return null;
  const next = Math.min(index + 1, times.length - 1);
  if (scene.frames[next]!.segment !== segment) return poseYaw[index]!;
  const span = times[next]! - times[index]!;
  const mix = span > 0 ? Math.max(0, Math.min(1, (t - times[index]!) / span)) : 0;
  return lerp(poseYaw[index]!, poseYaw[next]!, mix);
}

/**
 * Yaw averaged over a window around t, weighted towards t. It reads only the timeline, so a camera
 * built on it is still a pure function of time. The window stops at the shot's edges.
 */
function easedYaw(timeline: Timeline, t: number, segment: number, yaw: number): number {
  let sum = 0;
  let weights = 0;
  for (let step = -EASED_YAW_SAMPLES; step <= EASED_YAW_SAMPLES; step++) {
    const fraction = step / EASED_YAW_SAMPLES;
    const value = yawInShot(timeline, t + fraction * EASED_YAW_SECONDS, segment);
    if (value === null) continue;
    const weight = 1 - Math.abs(fraction);
    sum += value * weight;
    weights += weight;
  }
  return weights > 0 ? sum / weights : yaw;
}

/** World to ego: where a point of the shot's world sits relative to the car at this pose. */
export function worldToEgo(x: number, y: number, pose: EgoPose): { x: number; y: number } {
  const dx = x - pose.x;
  const dy = y - pose.y;
  const cos = Math.cos(pose.yaw);
  const sin = Math.sin(pose.yaw);
  return { x: dx * cos - dy * sin, y: dx * sin + dy * cos };
}

/**
 * The rigid transform that carries a world built in three.js axes (X = x, Z = -y) into the ego
 * frame, as a group's position and Y rotation.
 */
export function egoFrameTransform(pose: EgoPose): { x: number; z: number; rotationY: number } {
  const cos = Math.cos(pose.yaw);
  const sin = Math.sin(pose.yaw);
  return {
    x: -pose.x * cos + pose.y * sin,
    z: pose.x * sin + pose.y * cos,
    rotationY: pose.yaw,
  };
}

/** The static street for a shot, or null when there isn't one to build. */
export function worldForSegment(scene: SceneFile, segment: number): WorldSegment | null {
  const found = scene.world?.segments.find((entry) => entry.segment === segment);
  return found && found.road.length >= 2 ? found : null;
}

function lerp(a: number, b: number, mix: number): number {
  return a + (b - a) * mix;
}
