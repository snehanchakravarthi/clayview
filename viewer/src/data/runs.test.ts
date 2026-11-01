import { describe, expect, test } from "bun:test";
import { cleanRun, verge, type RunOptions } from "./runs";
import type { RoadStation, WorldPoint } from "./scene";
import { buildStreetLine, StreetIndex } from "./street";

const OPTIONS: RunOptions = {
  splitJump: 3,
  splitGap: 14,
  minLength: 4,
  clearance: 0.3,
  behindVerge: true,
  step: 3,
};

function road(walk = 0): StreetIndex {
  const stations: RoadStation[] = Array.from({ length: 120 }, (_, i) => ({
    s: i * 2,
    x: 0,
    y: i * 2,
    yaw: 0,
    left: -5,
    right: 5,
    measured: true,
    leftCurb: false,
    rightCurb: false,
    leftWalk: walk,
    rightWalk: walk,
    leftGrass: 0,
    rightGrass: 0,
  }));
  return new StreetIndex(buildStreetLine(stations)!);
}

function wall(offset: number, from: number, to: number): WorldPoint[] {
  const points: WorldPoint[] = [];
  for (let y = from; y <= to; y += 4) points.push([offset, y]);
  return points;
}

describe("cleanRun", () => {
  test("a straight wall comes back straight and evenly spaced", () => {
    const runs = cleanRun(wall(9, 20, 60), "right", road(), OPTIONS);
    expect(runs.length).toBe(1);
    const run = runs[0]!;
    expect(run.length).toBeCloseTo(40);
    for (const [x] of run.points) expect(x).toBeCloseTo(9);
    const gap = run.points[1]![1] - run.points[0]![1];
    expect(gap).toBeCloseTo(40 / Math.round(40 / 3));
  });

  test("noise is settled, so a wall doesn't zigzag", () => {
    const points = wall(9, 20, 80).map(([x, y], i): WorldPoint => [x + (i % 2 ? 0.8 : -0.8), y]);
    const [run] = cleanRun(points, "right", road(), { ...OPTIONS, splitJump: 5 });
    const xs = run!.points.map(([x]) => x);
    expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(1);
  });

  test("a jump starts a new run instead of drawing a diagonal across a gap", () => {
    const points = [...wall(9, 20, 44), ...wall(20, 48, 72)];
    expect(cleanRun(points, "right", road(), OPTIONS).length).toBe(2);
  });

  test("a long gap along the road splits a run", () => {
    const points = [...wall(9, 20, 40), ...wall(9, 80, 100)];
    expect(cleanRun(points, "right", road(), OPTIONS).length).toBe(2);
  });

  test("too short a run is noise", () => {
    expect(cleanRun(wall(9, 20, 22), "right", road(), OPTIONS)).toEqual([]);
  });

  test("nothing stands on the road", () => {
    const [run] = cleanRun(wall(3, 20, 60), "right", road(), OPTIONS);
    for (const [x] of run!.points) expect(x).toBeGreaterThanOrEqual(5.29);
  });

  test("a wall stays behind the sidewalk too", () => {
    const [run] = cleanRun(wall(6, 20, 60), "right", road(3), OPTIONS);
    for (const [x] of run!.points) expect(x).toBeGreaterThanOrEqual(8.29);
  });

  test("the left side mirrors the right", () => {
    const [run] = cleanRun(wall(-3, 20, 60), "left", road(), OPTIONS);
    for (const [x] of run!.points) expect(x).toBeLessThanOrEqual(-5.29);
  });

  test("points on the wrong side of the road are ignored", () => {
    expect(cleanRun(wall(9, 20, 60), "left", road(), OPTIONS)).toEqual([]);
  });

  test("points come back in road order however they arrived", () => {
    const [run] = cleanRun(wall(9, 20, 60).reverse(), "right", road(), OPTIONS);
    expect(run!.points[0]![1]).toBeLessThan(run!.points[run!.points.length - 1]![1]);
  });
});

describe("verge", () => {
  test("adds the sidewalk only when asked", () => {
    const { line } = road(3);
    expect(verge(line, 40, 1, 0.5, false)).toBeCloseTo(5.5);
    expect(verge(line, 40, 1, 0.5, true)).toBeCloseTo(8.5);
    expect(verge(line, 40, -1, 0.5, false)).toBeCloseTo(5.5);
  });
});
