import { describe, expect, test } from "bun:test";
import {
  buildTimeline,
  egoFrameTransform,
  lowerIndex,
  nearestAngle,
  parseScene,
  sampleEgo,
  sampleTrack,
  SCENE_VERSION,
  SceneError,
  worldForSegment,
  worldToEgo,
  type RoadStation,
  type SceneFile,
  type SceneFrame,
  type ScenePose,
} from "./scene";

function frame(index: number, objects: SceneFrame["objects"]): SceneFrame {
  return {
    frame: index,
    t: index * 0.1,
    segment: 0,
    ego: { speed: 10, yaw: 0, mount: 0, pose: [0, index, 0] },
    objects,
  };
}

function posed(index: number, pose: ScenePose, segment = 0): SceneFrame {
  return { ...frame(index, []), segment, ego: { speed: 10, yaw: 0, mount: 0, pose } };
}

function car(x: number, y: number, heading = 0): SceneFrame["objects"][number] {
  return { id: 1, class: "car", x, y, heading, confidence: 0.8, flags: [] };
}

function sceneFrom(frames: SceneFrame[]): SceneFile {
  const source = {
    path: "clip.mp4",
    width: 1920,
    height: 1080,
    fps: 10,
    duration: frames.length * 0.1,
    frameCount: frames.length,
    processedFrames: frames.length,
  };
  return {
    version: SCENE_VERSION,
    metadata: { source, camera: null, model: {}, egoMotion: { mode: "flow", coverage: 1 } },
    segments: [],
    frames,
  };
}

describe("parseScene", () => {
  test("names what is wrong", () => {
    expect(() => parseScene("{oops")).toThrow(SceneError);
    expect(() => parseScene("{}")).toThrow("isn't a clayview scene");
    const old = JSON.stringify({ ...sceneFrom([frame(0, [])]), version: 99 });
    expect(() => parseScene(old)).toThrow("version 99");
  });
});

describe("sampling", () => {
  test("interpolates position between frames", () => {
    const timeline = buildTimeline(sceneFrom([0, 1, 2].map((i) => frame(i, [car(i, 20 + i)]))));
    const sample = sampleTrack(timeline.tracks[0]!, 0.05);
    expect(sample?.x).toBeCloseTo(0.5);
    expect(sample?.y).toBeCloseTo(20.5);
  });

  test("turns heading the short way across the wrap", () => {
    const frames = [frame(0, [car(0, 20, 3.1)]), frame(1, [car(0, 20, -3.1)])];
    const timeline = buildTimeline(sceneFrom(frames));
    const middle = sampleTrack(timeline.tracks[0]!, 0.05)?.heading ?? 0;
    expect(Math.abs(Math.abs(middle) - Math.PI)).toBeLessThan(0.1);
  });

  test("fades in and out at the ends of a track", () => {
    const frames = Array.from({ length: 30 }, (_, i) => frame(i, [car(0, 20)]));
    const track = buildTimeline(sceneFrom(frames)).tracks[0]!;
    expect(sampleTrack(track, 0)?.fade).toBe(0);
    expect(sampleTrack(track, 1.5)?.fade).toBe(1);
    expect(sampleTrack(track, 2.9)?.fade).toBeCloseTo(0);
    expect(sampleTrack(track, 2.95)).toBeNull();
  });

  test("fade is a pure function of time, so scrubbing can't leave ghosts", () => {
    const frames = Array.from({ length: 30 }, (_, i) => frame(i, [car(0, 20)]));
    const track = buildTimeline(sceneFrom(frames)).tracks[0]!;
    const first = sampleTrack(track, 0.2)?.fade;
    sampleTrack(track, 2.5);
    expect(sampleTrack(track, 0.2)?.fade).toBe(first!);
  });

  test("ego distance integrates speed for the road scroll", () => {
    const timeline = buildTimeline(sceneFrom([0, 1, 2, 3].map((i) => frame(i, []))));
    expect(sampleEgo(timeline, 0.3).distance).toBeCloseTo(3);
    expect(sampleEgo(timeline, 0.15).distance).toBeCloseTo(1.5);
  });

  test("curvature stays zero when barely moving", () => {
    const frames = [0, 1, 2].map((i) => ({
      ...frame(i, []),
      ego: { speed: 0.5, yaw: 0.05, mount: 0, pose: [0, 0, 0] as ScenePose },
    }));
    expect(sampleEgo(buildTimeline(sceneFrom(frames)), 0.1).curvature).toBe(0);
  });
});

describe("ego pose", () => {
  test("interpolates between frames, so any playback speed is smooth", () => {
    const frames = [posed(0, [0, 0, 0]), posed(1, [2, 4, 0.2])];
    const { pose } = sampleEgo(buildTimeline(sceneFrom(frames)), 0.05);
    expect(pose.x).toBeCloseTo(1);
    expect(pose.y).toBeCloseTo(2);
    expect(pose.yaw).toBeCloseTo(0.1);
  });

  test("turns the short way when yaw wraps", () => {
    const frames = [posed(0, [0, 0, 3.1]), posed(1, [0, 1, -3.1])];
    const { yaw } = sampleEgo(buildTimeline(sceneFrom(frames)), 0.05).pose;
    expect(Math.abs(Math.abs(yaw) - Math.PI)).toBeLessThan(0.1);
  });

  test("holds at a cut rather than sliding across two unrelated worlds", () => {
    const frames = [posed(0, [50, 90, 1], 0), posed(1, [0, 0, 0], 1), posed(2, [0, 1, 0], 1)];
    const timeline = buildTimeline(sceneFrom(frames));
    const before = sampleEgo(timeline, 0.09);
    expect(before.segment).toBe(0);
    expect(before.pose.x).toBeCloseTo(50);
    const after = sampleEgo(timeline, 0.1);
    expect(after.segment).toBe(1);
    expect(after.pose.y).toBeCloseTo(0);
  });

  test("yaw doesn't accumulate across a cut", () => {
    const frames = [posed(0, [0, 0, 3.1], 0), posed(1, [0, 0, -3.1], 1)];
    const timeline = buildTimeline(sceneFrom(frames));
    expect(timeline.poseYaw[1]).toBeCloseTo(-3.1);
  });

  test("the camera's turn on its mount is sampled apart from the car's pose", () => {
    const frames = [0, 1, 2].map((i) => ({
      ...posed(i, [0, i, 0]),
      ego: { speed: 10, yaw: 0, mount: [0, 0.4, -0.2][i]!, pose: [0, i, 0] as ScenePose },
    }));
    const timeline = buildTimeline(sceneFrom(frames));
    const glance = sampleEgo(timeline, 0.05);
    expect(glance.mount).toBeCloseTo(0.2);
    expect(glance.pose.yaw).toBe(0);
    sampleEgo(timeline, 0.2);
    expect(sampleEgo(timeline, 0.05).mount).toBe(glance.mount);
  });

  test("the eased yaw lags a turn and stays a pure function of time", () => {
    const frames = Array.from({ length: 60 }, (_, i) => posed(i, [0, i, i < 30 ? 0 : 1]));
    const timeline = buildTimeline(sceneFrom(frames));
    const at = sampleEgo(timeline, 3.05).easedYaw;
    expect(at).toBeGreaterThan(0.1);
    expect(at).toBeLessThan(0.9);
    sampleEgo(timeline, 5);
    expect(sampleEgo(timeline, 3.05).easedYaw).toBe(at);
  });
});

describe("world transform", () => {
  test("a point dead ahead stays dead ahead at any heading", () => {
    const pose = { x: 12, y: -4, yaw: 0.7 };
    const ahead = worldToEgo(
      pose.x + Math.sin(pose.yaw) * 10,
      pose.y + Math.cos(pose.yaw) * 10,
      pose,
    );
    expect(ahead.x).toBeCloseTo(0);
    expect(ahead.y).toBeCloseTo(10);
  });

  test("a point on the right vector lands on ego's right", () => {
    const pose = { x: 0, y: 0, yaw: 1.2 };
    const right = worldToEgo(Math.cos(pose.yaw) * 3, -Math.sin(pose.yaw) * 3, pose);
    expect(right.x).toBeCloseTo(3);
    expect(right.y).toBeCloseTo(0);
  });

  test("the three.js transform agrees with the world-to-ego formula", () => {
    const pose = { x: 37.6, y: 103.4, yaw: 1.335 };
    const { x, z, rotationY } = egoFrameTransform(pose);
    const cos = Math.cos(rotationY);
    const sin = Math.sin(rotationY);
    for (const [wx, wy] of [
      [40, 110],
      [-5, 2],
      [300, 90],
    ] as const) {
      // Three's Y rotation, applied to the world point in three.js axes.
      const localX = wx;
      const localZ = -wy;
      const movedX = localX * cos + localZ * sin + x;
      const movedZ = -localX * sin + localZ * cos + z;
      const ego = worldToEgo(wx, wy, pose);
      expect(movedX).toBeCloseTo(ego.x);
      expect(-movedZ).toBeCloseTo(ego.y);
    }
  });
});

describe("worldForSegment", () => {
  const station = (s: number): RoadStation => ({
    s,
    x: 0,
    y: s,
    yaw: 0,
    left: -3,
    right: 3,
    measured: true,
    leftCurb: false,
    rightCurb: false,
    leftWalk: 0,
    rightWalk: 0,
    leftGrass: 0,
    rightGrass: 0,
  });
  const world = (segment: number, stations: number) => ({
    segment,
    road: Array.from({ length: stations }, (_, i) => station(i * 2)),
    markings: [],
    objects: [],
    buildings: [],
    barriers: [],
  });

  test("matches by segment, not by position in the list", () => {
    const scene = { ...sceneFrom([frame(0, [])]), world: { segments: [world(4, 3), world(2, 3)] } };
    expect(worldForSegment(scene, 2)?.segment).toBe(2);
    expect(worldForSegment(scene, 1)).toBeNull();
  });

  test("a null world or a stub of a road means no street", () => {
    expect(worldForSegment({ ...sceneFrom([frame(0, [])]), world: null }, 0)).toBeNull();
    const stub = { ...sceneFrom([frame(0, [])]), world: { segments: [world(0, 1)] } };
    expect(worldForSegment(stub, 0)).toBeNull();
  });
});

describe("helpers", () => {
  test("lowerIndex clamps at both ends", () => {
    const values = new Float64Array([0, 1, 2]);
    expect(lowerIndex(values, -1)).toBe(0);
    expect(lowerIndex(values, 1.5)).toBe(1);
    expect(lowerIndex(values, 9)).toBe(2);
  });

  test("nearestAngle stays within half a turn of the reference", () => {
    expect(nearestAngle(-3.1, 3.1)).toBeCloseTo(-3.1 + Math.PI * 2);
    expect(nearestAngle(0.2, 0)).toBeCloseTo(0.2);
  });
});
