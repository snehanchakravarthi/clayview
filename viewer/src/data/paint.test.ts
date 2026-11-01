import { describe, expect, test } from "bun:test";
import {
  arrowRing,
  cleanMarking,
  cleanMarkings,
  ringArea,
  ringPerimeter,
  simplifyLine,
  simplifyRing,
  type PaintShape,
} from "./paint";
import type { RoadMarking, RoadStation, WorldPoint } from "./scene";
import { buildStreetLine, StreetIndex } from "./street";

function road(): StreetIndex {
  const stations: RoadStation[] = Array.from({ length: 120 }, (_, i) => ({
    s: i * 2,
    x: 0,
    y: i * 2,
    yaw: 0,
    left: -8,
    right: 8,
    measured: true,
    leftCurb: false,
    rightCurb: false,
    leftWalk: 0,
    rightWalk: 0,
    leftGrass: 0,
    rightGrass: 0,
  }));
  return new StreetIndex(buildStreetLine(stations)!);
}

/** A box in road coordinates, with the staircase edge the pixel grid gives real polygons. */
function noisyBox(x0: number, x1: number, y0: number, y1: number): WorldPoint[] {
  const ring: WorldPoint[] = [];
  const stepping = (a: number, b: number): number[] => {
    const steps = Math.max(Math.round(Math.abs(b - a) / 0.4), 1);
    return Array.from({ length: steps }, (_, i) => a + ((b - a) * i) / steps);
  };
  stepping(y0, y1).forEach((y, i) => ring.push([x1 + (i % 2) * 0.05, y]));
  stepping(x1, x0).forEach((x, i) => ring.push([x, y1 + (i % 2) * 0.05]));
  stepping(y1, y0).forEach((y, i) => ring.push([x0 - (i % 2) * 0.05, y]));
  stepping(x0, x1).forEach((x, i) => ring.push([x, y0 - (i % 2) * 0.05]));
  return ring;
}

function line(points: WorldPoint[]): RoadMarking {
  return { kind: "line", points };
}

function kinds(shapes: PaintShape[]): string[] {
  return shapes.map((shape) => shape.kind);
}

function bounds(shape: PaintShape): { x0: number; x1: number; y0: number; y1: number } {
  const xs = shape.ring.map(([x]) => x);
  const ys = shape.ring.map(([, y]) => y);
  return {
    x0: Math.min(...xs),
    x1: Math.max(...xs),
    y0: Math.min(...ys),
    y1: Math.max(...ys),
  };
}

describe("cleanMarking", () => {
  const index = road();

  test("a lane dash comes out as one clean straight stroke", () => {
    const shapes = cleanMarking(line(noisyBox(2, 2.18, 40, 44)), index);
    expect(kinds(shapes)).toEqual(["stroke"]);
    const box = bounds(shapes[0]!);
    expect(box.x1 - box.x0).toBeLessThan(0.3);
    expect(box.y1 - box.y0).toBeGreaterThan(3.8);
    expect((box.x0 + box.x1) / 2).toBeCloseTo(2.09, 0);
  });

  test("a wobbly short line is straightened, because paint is laid by machine", () => {
    const wobble: WorldPoint[] = [];
    for (let y = 40; y <= 48; y += 1) wobble.push([2 + (y % 2) * 0.1, y]);
    for (let y = 48; y >= 40; y -= 1) wobble.push([2.15 + (y % 2) * 0.1, y]);
    const [stroke] = cleanMarking(line(wobble), index);
    const box = bounds(stroke!);
    expect(box.x1 - box.x0).toBeLessThan(0.34);
  });

  test("a line never comes out wider than paint is", () => {
    const [stroke] = cleanMarking(line(noisyBox(2, 2.5, 40, 56)), index);
    const box = bounds(stroke!);
    expect(box.x1 - box.x0).toBeLessThan(0.3);
  });

  test("a stop line stays a square bar across the road", () => {
    const shapes = cleanMarking(line(noisyBox(-3, 3, 60, 60.4)), index);
    expect(kinds(shapes)).toEqual(["bar"]);
    const box = bounds(shapes[0]!);
    expect(box.x1 - box.x0).toBeGreaterThan(5.5);
    expect(box.y1 - box.y0).toBeLessThanOrEqual(0.6);
  });

  test("a long smear collapses to the line that caused it", () => {
    const shapes = cleanMarking(line(noisyBox(3, 5, 40, 70)), index);
    expect(kinds(shapes)).toEqual(["stroke"]);
    const box = bounds(shapes[0]!);
    expect(box.x1 - box.x0).toBeLessThan(0.4);
    expect((box.x0 + box.x1) / 2).toBeGreaterThan(3.5);
    expect((box.x0 + box.x1) / 2).toBeLessThan(4.5);
  });

  test("a short fat smear is dropped, not painted", () => {
    expect(cleanMarking(line(noisyBox(1, 4, 40, 43)), index)).toEqual([]);
  });

  test("a compact lump isn't believed as paint", () => {
    expect(cleanMarking(line(noisyBox(1, 2, 40, 41)), index)).toEqual([]);
  });

  test("a crosswalk becomes evenly spaced stripes along the road", () => {
    const shapes = cleanMarking(
      { kind: "crosswalk", points: noisyBox(-3, 3, 60, 63) },
      index,
    );
    expect(shapes.length).toBe(6);
    expect(new Set(kinds(shapes))).toEqual(new Set(["zebra"]));
    const first = bounds(shapes[0]!);
    const second = bounds(shapes[1]!);
    expect(second.x0 - first.x0).toBeCloseTo(1);
    expect(first.y1 - first.y0).toBeGreaterThan(2.5);
    expect(first.x1 - first.x0).toBeCloseTo(0.52);
  });

  test("a one-stripe crosswalk fragment still gets a believable length", () => {
    const [stripe] = cleanMarking({ kind: "crosswalk", points: noisyBox(0, 0.5, 60, 61) }, index);
    const box = bounds(stripe!);
    expect(box.y1 - box.y0).toBeGreaterThan(2.59);
  });

  test("an arrow-shaped polygon is redrawn as a clean arrow pointing the same way", () => {
    // A shaft with a head at the far end.
    const shape: WorldPoint[] = [
      [1.9, 40],
      [2.1, 40],
      [2.1, 43.2],
      [2.6, 43],
      [2, 45],
      [1.4, 43],
      [1.9, 43.2],
    ];
    const [arrow] = cleanMarking(line(shape), index);
    expect(arrow?.kind).toBe("arrow");
    const tip = arrow!.ring.reduce((best, point) => (point[1] > best[1] ? point : best));
    expect(tip[0]).toBeCloseTo(2, 0);
    const width = bounds(arrow!);
    expect(width.y1).toBeGreaterThan(44);
  });

  test("an arrow with its head at the near end points back", () => {
    const shape: WorldPoint[] = [
      [1.9, 45],
      [2.1, 45],
      [2.1, 41.8],
      [2.6, 42],
      [2, 40],
      [1.4, 42],
      [1.9, 41.8],
    ];
    const [arrow] = cleanMarking(line(shape), index);
    expect(arrow?.kind).toBe("arrow");
    const tips = [...arrow!.ring].sort((a, b) => a[1] - b[1]);
    expect(Math.abs(tips[0]![0] - 2)).toBeLessThan(0.2);
  });

  test("paint found off the road is dropped", () => {
    expect(cleanMarking(line(noisyBox(14, 14.2, 40, 44)), index)).toEqual([]);
  });

  test("a thin zigzag keeps its shape", () => {
    const zigzag: WorldPoint[] = [];
    for (let i = 0; i < 6; i++) zigzag.push([3 + (i % 2) * 1.2, 40 + i * 0.9]);
    for (let i = 5; i >= 0; i--) zigzag.push([3.22 + (i % 2) * 1.2, 40 + i * 0.9]);
    const shapes = cleanMarking(line(zigzag), index);
    expect(kinds(shapes)).toEqual(["figure"]);
    expect(shapes[0]!.ring.length).toBeGreaterThanOrEqual(8);
  });

  test("a degenerate polygon doesn't throw", () => {
    expect(cleanMarking(line([[0, 0], [1, 1]]), index)).toEqual([]);
    expect(
      cleanMarking(line([[2, 40], [2, 40], [2, 40], [2, 40]]), index),
    ).toEqual([]);
  });

  test("cleanMarkings keeps going past shapes it rejects", () => {
    const shapes = cleanMarkings(
      [line(noisyBox(14, 14.2, 40, 44)), line(noisyBox(2, 2.18, 40, 44))],
      index,
    );
    expect(shapes.length).toBe(1);
  });
});

describe("geometry helpers", () => {
  test("simplifyLine drops points on a straight run and keeps corners", () => {
    const points: [number, number][] = [
      [0, 0],
      [1, 0.01],
      [2, 0],
      [2, 2],
    ];
    expect(simplifyLine(points, 0.1)).toEqual([
      [0, 0],
      [2, 0],
      [2, 2],
    ]);
  });

  test("simplifyRing takes the stairs off a rectangle", () => {
    const ring = noisyBox(0, 0.2, 0, 4);
    const simple = simplifyRing(ring, 0.14);
    expect(simple.length).toBeLessThan(ring.length / 2);
    expect(ringArea(simple)).toBeGreaterThan(0.5);
  });

  test("ring measures", () => {
    const square: [number, number][] = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ];
    expect(ringArea(square)).toBeCloseTo(4);
    expect(ringPerimeter(square)).toBeCloseTo(8);
  });

  test("an arrow's head is wider than its shaft", () => {
    const ring = arrowRing(10, 0, 5, 1);
    const across = ring.map(([, d]) => Math.abs(d));
    expect(Math.max(...across)).toBeGreaterThan(0.4);
    expect(across[0]).toBeLessThan(0.2);
  });
});
