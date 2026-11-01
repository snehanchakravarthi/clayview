/** The scene file contract, and fast sampling of it at any time. */

export const SCENE_VERSION = 1;

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

export interface SceneFrame {
  frame: number;
  t: number;
  segment: number;
  ego: { speed: number; yaw: number };
  objects: SceneObject[];
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

export interface EgoSample {
  speed: number;
  distance: number;
  curvature: number;
  segment: number;
}

export class SceneError extends Error {}

const FADE_SECONDS = 0.35;
const MAX_CURVATURE = 1 / 70;
const CURVATURE_WINDOW_SECONDS = 1.2;
const MIN_CURVING_SPEED = 2;

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

  const grouped = new Map<number, { t: number; object: SceneObject }[]>();
  for (let index = 0; index < count; index++) {
    const frame = frames[index]!;
    times[index] = frame.t;
    speed[index] = frame.ego.speed;
    density[index] = frame.objects.length;
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
  return {
    speed: lerp(speed[index]!, speed[next]!, mix),
    distance: lerp(distance[index]!, distance[next]!, mix),
    curvature: lerp(curvature[index]!, curvature[next]!, mix),
    segment: scene.frames[index]!.segment,
  };
}

function lerp(a: number, b: number, mix: number): number {
  return a + (b - a) * mix;
}
