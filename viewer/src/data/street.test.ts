import { describe, expect, test } from "bun:test";
import type { RoadStation } from "./scene";
import { buildStreetLine, gaussian, median, seeded, stationAt, StreetIndex } from "./street";

function station(s: number, x: number, y: number, yaw: number): RoadStation {
  return {
    s,
    x,
    y,
    yaw,
    left: -5,
    right: 5,
    measured: true,
    leftCurb: true,
    rightCurb: true,
    leftWalk: 3,
    rightWalk: 3,
    leftGrass: 2,
    rightGrass: 2,
  };
}

function straight(count: number, patch: Partial<RoadStation> = {}): RoadStation[] {
  return Array.from({ length: count }, (_, i) => ({ ...station(i * 2, 0, i * 2, 0), ...patch }));
}

/** A path made of a straight, a hard bend of `radius`, and another straight. */
function bend(radius: number, turn: number): RoadStation[] {
  const road: RoadStation[] = [];
  let x = 0;
  let y = 0;
  let yaw = 0;
  const step = 2;
  const add = (): void => {
    road.push(station(road.length * step, x, y, yaw));
    x += Math.sin(yaw) * step;
    y += Math.cos(yaw) * step;
  };
  for (let i = 0; i < 30; i++) add();
  const arc = Math.round((turn * radius) / step);
  for (let i = 0; i < arc; i++) {
    add();
    yaw += step / radius;
  }
  for (let i = 0; i < 30; i++) add();
  return road;
}

/** Fails if the outermost line on a side ever runs backwards relative to the road. */
function worstStep(road: RoadStation[], sign: 1 | -1): number {
  const line = buildStreetLine(road)!;
  let worst = Infinity;
  for (let i = 0; i + 1 < line.count; i++) {
    const offset = (index: number): number =>
      sign > 0
        ? line.right[index]! + line.walkRight[index]! + line.grassRight[index]!
        : line.left[index]! - line.walkLeft[index]! - line.grassLeft[index]!;
    const ax = line.x[i]! + line.rx[i]! * offset(i);
    const ay = line.y[i]! + line.ry[i]! * offset(i);
    const bx = line.x[i + 1]! + line.rx[i + 1]! * offset(i + 1);
    const by = line.y[i + 1]! + line.ry[i + 1]! * offset(i + 1);
    worst = Math.min(worst, (bx - ax) * line.tx[i + 1]! + (by - ay) * line.ty[i + 1]!);
  }
  return worst;
}

describe("buildStreetLine", () => {
  test("keeps the road's width on a straight", () => {
    const line = buildStreetLine(straight(100))!;
    const middle = line.firstReal + 50;
    expect(line.left[middle]).toBeCloseTo(-5);
    expect(line.right[middle]).toBeCloseTo(5);
    expect(line.walkRight[middle]).toBeCloseTo(3);
    expect(line.x[middle]).toBeCloseTo(0);
  });

  test("offsets point right of travel, as the data does", () => {
    const line = buildStreetLine(straight(40))!;
    const middle = line.firstReal + 20;
    expect(line.rx[middle]).toBeCloseTo(1);
    expect(line.ry[middle]).toBeCloseTo(0);
  });

  test("s is zero at the first real station", () => {
    const line = buildStreetLine(straight(40))!;
    expect(line.s[line.firstReal]).toBeCloseTo(0);
    expect(line.s[0]!).toBeLessThan(0);
  });

  test("carries the road on past both ends, fading away", () => {
    const line = buildStreetLine(straight(40))!;
    expect(line.y[line.count - 1]!).toBeGreaterThan(78 + 150);
    expect(line.y[0]!).toBeLessThan(-40);
    expect(line.presence[line.firstReal + 5]).toBe(1);
    expect(line.presence[line.count - 1]).toBeCloseTo(0);
    expect(line.presence[0]).toBeCloseTo(0);
  });

  test("the run-out continues straight, however the last station's yaw jitters", () => {
    const road = straight(40);
    road[39]!.yaw = 0.8;
    const line = buildStreetLine(road)!;
    expect(Math.abs(line.x[line.count - 1]!)).toBeLessThan(40);
  });

  test("takes single-station spikes out of the edges", () => {
    const road = straight(60);
    road[30]!.left = -13;
    const line = buildStreetLine(road)!;
    expect(line.left[line.firstReal + 30]!).toBeGreaterThan(-5.6);
  });

  test("unmeasured stations follow their measured neighbours", () => {
    const road = straight(60, { left: -4 });
    for (let i = 26; i < 34; i++) Object.assign(road[i]!, { left: -9, measured: false });
    const line = buildStreetLine(road)!;
    expect(line.left[line.firstReal + 30]!).toBeGreaterThan(-7.5);
    expect(line.trust[line.firstReal + 30]!).toBeLessThan(0.5);
    expect(line.trust[line.firstReal + 5]).toBeCloseTo(1);
  });

  test("the verge never folds over on the inside of a hard bend", () => {
    for (const radius of [4, 7, 12, 20]) {
      const road = bend(radius, Math.PI / 2);
      expect(worstStep(road, 1)).toBeGreaterThan(0);
      expect(worstStep(road, -1)).toBeGreaterThan(0);
    }
  });

  test("the road survives a U-turn", () => {
    const road = bend(5, Math.PI);
    const line = buildStreetLine(road)!;
    expect(worstStep(road, 1)).toBeGreaterThan(0);
    expect(worstStep(road, -1)).toBeGreaterThan(0);
    for (let i = 0; i < line.count; i++) expect(line.right[i]!).toBeGreaterThan(1);
  });

  test("the outside of a bend keeps its full width", () => {
    const road = bend(15, Math.PI / 2);
    const line = buildStreetLine(road)!;
    const apex = line.firstReal + 30 + Math.round((15 * Math.PI) / 4 / 2);
    expect(line.left[apex]!).toBeLessThan(-4.5);
    expect(line.right[apex]!).toBeLessThan(5.01);
  });

  test("a road too short to smooth is refused", () => {
    expect(buildStreetLine([station(0, 0, 0, 0)])).toBeNull();
  });
});

describe("StreetIndex", () => {
  test("round-trips a world point through road coordinates", () => {
    const index = new StreetIndex(buildStreetLine(bend(25, 1))!);
    for (const [x, y] of [
      [3, 20],
      [-6, 70],
      [20, 90],
    ] as const) {
      const place = index.place(x, y);
      const [backX, backY] = index.point(place.s, place.d);
      expect(Math.hypot(backX - x, backY - y)).toBeLessThan(0.4);
    }
  });

  test("right of the road is positive", () => {
    const index = new StreetIndex(buildStreetLine(straight(50))!);
    expect(index.place(4, 30).d).toBeCloseTo(4);
    expect(index.place(-4, 30).d).toBeCloseTo(-4);
    expect(index.place(0, 30).s).toBeCloseTo(30);
  });

  test("finds the road from far off it", () => {
    const index = new StreetIndex(buildStreetLine(straight(50))!);
    expect(index.place(400, 30).d).toBeCloseTo(400);
  });
});

describe("helpers", () => {
  test("gaussian leaves a constant alone and flattens a spike", () => {
    const flat = gaussian([2, 2, 2, 2], 1.5);
    expect(flat[1]).toBeCloseTo(2);
    expect(gaussian([0, 0, 9, 0, 0], 1.5)[2]!).toBeLessThan(5);
  });

  test("gaussian weights let measured samples lead", () => {
    const smooth = gaussian([1, 9, 1], 1, [1, 0, 1]);
    expect(smooth[1]).toBeCloseTo(1);
  });

  test("median ignores one wild value", () => {
    expect(median([1, 1, 50, 1, 1], 1)[2]).toBe(1);
  });

  test("stationAt finds the station at or before s", () => {
    const line = buildStreetLine(straight(40))!;
    const index = stationAt(line, 10.5);
    expect(line.s[index]!).toBeLessThanOrEqual(10.5);
    expect(line.s[index + 1]!).toBeGreaterThan(10.5);
  });

  test("seeded is stable and spread out", () => {
    expect(seeded(3.2, 8)).toBe(seeded(3.2, 8));
    const values = Array.from({ length: 200 }, (_, i) => seeded(i * 1.7, i * 0.3));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    expect(mean).toBeGreaterThan(0.4);
    expect(mean).toBeLessThan(0.6);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
  });
});
