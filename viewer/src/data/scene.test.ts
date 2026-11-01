import { describe, expect, test } from "bun:test";
import {
  buildTimeline,
  lowerIndex,
  nearestAngle,
  parseScene,
  sampleEgo,
  sampleTrack,
  SceneError,
  type SceneFile,
  type SceneFrame,
} from "./scene";

function frame(index: number, objects: SceneFrame["objects"]): SceneFrame {
  return { frame: index, t: index * 0.1, segment: 0, ego: { speed: 10, yaw: 0 }, objects };
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
    version: 1,
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
      ego: { speed: 0.5, yaw: 0.05 },
    }));
    expect(sampleEgo(buildTimeline(sceneFrom(frames)), 0.1).curvature).toBe(0);
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
