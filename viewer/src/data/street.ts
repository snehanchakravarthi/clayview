/**
 * Turns the road stations of a shot into a clean line to build on: smooth, free of folds on the
 * inside of bends, carried on past both ends of the clip, and able to place any world point
 * relative to the road. Pure numbers, no three.js, so it can be tested on its own.
 */

import type { RoadStation, WorldPoint } from "./scene";

const STATION_M = 2;
/** The clip stops when the car does; the road ahead of it has to go on into the fog. */
const EXTEND_AHEAD_M = 200;
const EXTEND_BEHIND_M = 60;
const HEADING_SPAN = 8;
const WIDTH_SPAN = 10;

const CENTRE_SIGMA = 2.2;
const TANGENT_SIGMA = 1.2;
const EDGE_MEDIAN = 3;
const EDGE_SIGMA = 2.4;
const WIDTH_MEDIAN = 3;
const WIDTH_SIGMA = 2.2;
const FLAG_MEDIAN = 2;
const FLAG_SIGMA = 1.4;
const TRUST_SIGMA = 2.5;
const FOLD_MARGIN = 0.68;
const FOLD_SPREAD = 5;
const MIN_ROAD_SIDE = 1.2;
const MIN_SHOWN_WIDTH = 0.3;
const GRID_M = 18;

/** The road, sampled at stations. Offsets follow the data's sign: left negative, right positive. */
export interface StreetLine {
  count: number;
  /** Distance along the line; 0 is the clip's first station, so it runs negative behind it. */
  s: Float32Array;
  x: Float32Array;
  y: Float32Array;
  /** Unit vector along the line. */
  tx: Float32Array;
  ty: Float32Array;
  /** Unit vector to the right of the line. */
  rx: Float32Array;
  ry: Float32Array;
  left: Float32Array;
  right: Float32Array;
  curbLeft: Float32Array;
  curbRight: Float32Array;
  walkLeft: Float32Array;
  walkRight: Float32Array;
  grassLeft: Float32Array;
  grassRight: Float32Array;
  /** 1 where the edges were measured, falling towards 0 where they were guessed. */
  trust: Float32Array;
  /** 1 on the real road, falling to 0 along the extension past either end. */
  presence: Float32Array;
  /** Stations of the real road inside these arrays. */
  firstReal: number;
  lastReal: number;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Gaussian blur along a series. Ends clamp, and optional weights let good samples lead. */
export function gaussian(
  values: ArrayLike<number>,
  sigma: number,
  weights?: ArrayLike<number>,
): Float32Array {
  const count = values.length;
  const reach = Math.ceil(sigma * 3);
  const kernel = Array.from({ length: reach * 2 + 1 }, (_, index) =>
    Math.exp(-((index - reach) ** 2) / (2 * sigma * sigma)),
  );
  const result = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    let sum = 0;
    let total = 0;
    for (let step = -reach; step <= reach; step++) {
      const at = clamp(index + step, 0, count - 1);
      const weight = kernel[step + reach]! * (weights ? weights[at]! : 1);
      sum += values[at]! * weight;
      total += weight;
    }
    result[index] = total > 0 ? sum / total : values[index]!;
  }
  return result;
}

/** Median over a window of `reach` on each side. Takes out single-station spikes. */
export function median(values: ArrayLike<number>, reach: number): Float32Array {
  const count = values.length;
  const result = new Float32Array(count);
  const window: number[] = [];
  for (let index = 0; index < count; index++) {
    window.length = 0;
    for (let at = Math.max(index - reach, 0); at <= Math.min(index + reach, count - 1); at++) {
      window.push(values[at]!);
    }
    window.sort((a, b) => a - b);
    result[index] = window[window.length >> 1]!;
  }
  return result;
}

/** The lowest value in reach of each index. */
function erode(values: ArrayLike<number>, reach: number): Float32Array {
  const count = values.length;
  const result = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    let low = Infinity;
    for (let at = Math.max(index - reach, 0); at <= Math.min(index + reach, count - 1); at++) {
      low = Math.min(low, values[at]!);
    }
    result[index] = low;
  }
  return result;
}

interface Raw {
  x: number[];
  y: number[];
  rx: number[];
  ry: number[];
  station: (RoadStation | null)[];
  presence: number[];
}

function meanHeading(road: readonly RoadStation[], from: number, to: number): [number, number] {
  let x = 0;
  let y = 0;
  for (let index = from; index <= to; index++) {
    x += Math.sin(road[index]!.yaw);
    y += Math.cos(road[index]!.yaw);
  }
  const length = Math.hypot(x, y) || 1;
  return [x / length, y / length];
}

/** Stations with straight run-outs on both ends, so the line can be smoothed right up to them. */
function withRunOuts(road: readonly RoadStation[]): Raw {
  const raw: Raw = { x: [], y: [], rx: [], ry: [], station: [], presence: [] };
  const last = road.length - 1;
  const push = (x: number, y: number, yaw: number, at: RoadStation | null, seen: number): void => {
    raw.x.push(x);
    raw.y.push(y);
    raw.rx.push(Math.cos(yaw));
    raw.ry.push(-Math.sin(yaw));
    raw.station.push(at);
    raw.presence.push(seen);
  };

  const [backX, backY] = meanHeading(road, 0, Math.min(HEADING_SPAN, last));
  const backYaw = Math.atan2(backX, backY);
  const behind = Math.round(EXTEND_BEHIND_M / STATION_M);
  for (let step = behind; step >= 1; step--) {
    const distance = step * STATION_M;
    push(
      road[0]!.x - backX * distance,
      road[0]!.y - backY * distance,
      backYaw,
      null,
      1 - step / behind,
    );
  }
  for (const station of road) push(station.x, station.y, station.yaw, station, 1);
  const [aheadX, aheadY] = meanHeading(road, Math.max(last - HEADING_SPAN, 0), last);
  const aheadYaw = Math.atan2(aheadX, aheadY);
  const ahead = Math.round(EXTEND_AHEAD_M / STATION_M);
  for (let step = 1; step <= ahead; step++) {
    const distance = step * STATION_M;
    push(
      road[last]!.x + aheadX * distance,
      road[last]!.y + aheadY * distance,
      aheadYaw,
      null,
      1 - step / ahead,
    );
  }
  return raw;
}

/** One value per station, with run-outs holding the average of the road they carry on from. */
function column(
  road: readonly RoadStation[],
  raw: Raw,
  read: (station: RoadStation) => number,
): Float32Array {
  const result = new Float32Array(raw.station.length);
  const mean = (from: number, to: number): number => {
    let sum = 0;
    for (let index = from; index <= to; index++) sum += read(road[index]!);
    return sum / (to - from + 1);
  };
  const last = road.length - 1;
  const start = mean(0, Math.min(WIDTH_SPAN, last));
  const end = mean(Math.max(last - WIDTH_SPAN, 0), last);
  let seenRoad = false;
  raw.station.forEach((station, index) => {
    if (station) {
      seenRoad = true;
      result[index] = read(station);
    } else {
      result[index] = seenRoad ? end : start;
    }
  });
  return result;
}

/**
 * Keep an outward distance from folding back on itself: a point offset from a line only moves
 * forward while the offset stays inside the bend's radius. Returns the most each side can reach.
 */
function foldLimits(line: {
  count: number;
  x: Float32Array;
  y: Float32Array;
  tx: Float32Array;
  ty: Float32Array;
  rx: Float32Array;
  ry: Float32Array;
}): { left: Float32Array; right: Float32Array } {
  const { count, x, y, tx, ty, rx, ry } = line;
  const left = new Float32Array(count).fill(1e4);
  const right = new Float32Array(count).fill(1e4);
  const limit = (index: number, along: number, lean: number): void => {
    // `lean` is how far this station's right vector tips into the neighbour's direction of travel.
    if (lean > 1e-5) right[index] = Math.min(right[index]!, along / lean);
    if (lean < -1e-5) left[index] = Math.min(left[index]!, along / -lean);
  };
  for (let index = 0; index + 1 < count; index++) {
    const dx = x[index + 1]! - x[index]!;
    const dy = y[index + 1]! - y[index]!;
    const forward = (dx * tx[index + 1]! + dy * ty[index + 1]!) * FOLD_MARGIN;
    limit(index, forward, rx[index]! * tx[index + 1]! + ry[index]! * ty[index + 1]!);
    const backward = (dx * tx[index]! + dy * ty[index]!) * FOLD_MARGIN;
    limit(index + 1, backward, -(rx[index + 1]! * tx[index]! + ry[index + 1]! * ty[index]!));
  }
  return { left, right };
}

/** Spread hard limits out so a clamp eases in along the road rather than notching it. */
function softLimit(limit: Float32Array): Float32Array {
  const eased = gaussian(erode(limit, FOLD_SPREAD), FOLD_SPREAD / 2);
  for (let index = 0; index < limit.length; index++) {
    eased[index] = Math.min(eased[index]!, limit[index]!);
  }
  return eased;
}

function flagColumn(
  road: readonly RoadStation[],
  raw: Raw,
  read: (station: RoadStation) => boolean,
): Float32Array {
  const flags = column(road, raw, (station) => (read(station) ? 1 : 0));
  const smooth = gaussian(median(flags, FLAG_MEDIAN), FLAG_SIGMA);
  return smooth.map((value) => smoothstep(0.32, 0.68, value));
}

function widthColumn(
  road: readonly RoadStation[],
  raw: Raw,
  read: (station: RoadStation) => number,
): Float32Array {
  const smooth = gaussian(median(column(road, raw, read), WIDTH_MEDIAN), WIDTH_SIGMA);
  return smooth.map((value) => (value < MIN_SHOWN_WIDTH ? 0 : value));
}

/** Null when there's too little road to build. */
export function buildStreetLine(road: readonly RoadStation[]): StreetLine | null {
  if (road.length < 2) return null;
  const raw = withRunOuts(road);
  const count = raw.x.length;
  const firstReal = raw.station.findIndex((station) => station !== null);
  const lastReal = firstReal + road.length - 1;

  const x = gaussian(raw.x, CENTRE_SIGMA);
  const y = gaussian(raw.y, CENTRE_SIGMA);
  const tx = new Float32Array(count);
  const ty = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const before = Math.max(index - 1, 0);
    const after = Math.min(index + 1, count - 1);
    tx[index] = x[after]! - x[before]!;
    ty[index] = y[after]! - y[before]!;
  }
  const smoothX = gaussian(tx, TANGENT_SIGMA);
  const smoothY = gaussian(ty, TANGENT_SIGMA);
  const rx = new Float32Array(count);
  const ry = new Float32Array(count);
  const s = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const length = Math.hypot(smoothX[index]!, smoothY[index]!) || 1;
    tx[index] = smoothX[index]! / length;
    ty[index] = smoothY[index]! / length;
    rx[index] = ty[index]!;
    ry[index] = -tx[index]!;
    if (index > 0) {
      s[index] = s[index - 1]! + Math.hypot(x[index]! - x[index - 1]!, y[index]! - y[index - 1]!);
    }
  }
  const zero = s[firstReal]!;
  for (let index = 0; index < count; index++) s[index] = s[index]! - zero;

  // Edges are real places. Carry them across onto the smoothed line, then settle their noise.
  const leftRaw = column(road, raw, (station) => station.left);
  const rightRaw = column(road, raw, (station) => station.right);
  const reproject = (values: Float32Array): Float32Array =>
    values.map((offset, index) => {
      const edgeX = raw.x[index]! + raw.rx[index]! * offset;
      const edgeY = raw.y[index]! + raw.ry[index]! * offset;
      return (edgeX - x[index]!) * rx[index]! + (edgeY - y[index]!) * ry[index]!;
    });
  const measured = column(road, raw, (station) => (station.measured ? 1 : 0.15));
  const left = gaussian(median(reproject(leftRaw), EDGE_MEDIAN), EDGE_SIGMA, measured);
  const right = gaussian(median(reproject(rightRaw), EDGE_MEDIAN), EDGE_SIGMA, measured);

  let curbLeft = flagColumn(road, raw, (station) => station.leftCurb);
  let curbRight = flagColumn(road, raw, (station) => station.rightCurb);
  const walkLeft = widthColumn(road, raw, (station) => station.leftWalk);
  const walkRight = widthColumn(road, raw, (station) => station.rightWalk);
  const grassLeft = widthColumn(road, raw, (station) => station.leftGrass);
  const grassRight = widthColumn(road, raw, (station) => station.rightGrass);
  const trust = gaussian(measured, TRUST_SIGMA).map((value) => clamp((value - 0.15) / 0.85, 0, 1));
  const presence = Float32Array.from(raw.presence, (value) => smoothstep(0, 1, value));
  curbLeft = curbLeft.map((value, index) => value * smoothstep(0.55, 1, raw.presence[index]!));
  curbRight = curbRight.map((value, index) => value * smoothstep(0.55, 1, raw.presence[index]!));

  // Inside a bend the whole verge has to fit. Give up grass first, then sidewalk, then road.
  const limits = foldLimits({ count, x, y, tx, ty, rx, ry });
  const reachLeft = softLimit(limits.left);
  const reachRight = softLimit(limits.right);
  for (let index = 0; index < count; index++) {
    const roomLeft = Math.max(reachLeft[index]!, MIN_ROAD_SIDE);
    left[index] = Math.max(left[index]!, -roomLeft);
    if (left[index]! > -MIN_ROAD_SIDE) left[index] = -MIN_ROAD_SIDE;
    const spareLeft = Math.max(roomLeft + left[index]!, 0);
    walkLeft[index] = Math.min(walkLeft[index]!, spareLeft);
    grassLeft[index] = Math.min(grassLeft[index]!, spareLeft - walkLeft[index]!);

    const roomRight = Math.max(reachRight[index]!, MIN_ROAD_SIDE);
    right[index] = Math.min(right[index]!, roomRight);
    if (right[index]! < MIN_ROAD_SIDE) right[index] = MIN_ROAD_SIDE;
    const spareRight = Math.max(roomRight - right[index]!, 0);
    walkRight[index] = Math.min(walkRight[index]!, spareRight);
    grassRight[index] = Math.min(grassRight[index]!, spareRight - walkRight[index]!);
  }

  return {
    count,
    s,
    x,
    y,
    tx,
    ty,
    rx,
    ry,
    left,
    right,
    curbLeft,
    curbRight,
    walkLeft,
    walkRight,
    grassLeft,
    grassRight,
    trust,
    presence,
    firstReal,
    lastReal,
  };
}

/** Where a world point sits on the road. */
export interface RoadPlace {
  station: number;
  s: number;
  /** Signed offset along the right vector. */
  d: number;
}

/** Finds the nearest station to a world point, quickly enough to ask for every polygon vertex. */
export class StreetIndex {
  private readonly cells = new Map<string, number[]>();

  constructor(readonly line: StreetLine) {
    for (let index = 0; index < line.count; index++) {
      const key = StreetIndex.key(line.x[index]!, line.y[index]!);
      const cell = this.cells.get(key);
      if (cell) cell.push(index);
      else this.cells.set(key, [index]);
    }
  }

  private static key(x: number, y: number): string {
    return `${Math.floor(x / GRID_M)}:${Math.floor(y / GRID_M)}`;
  }

  nearest(x: number, y: number): number {
    const { line } = this;
    const cellX = Math.floor(x / GRID_M);
    const cellY = Math.floor(y / GRID_M);
    let best = -1;
    let bestDistance = Infinity;
    const visit = (index: number): void => {
      const distance = (line.x[index]! - x) ** 2 + (line.y[index]! - y) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    };
    for (let ring = 0; ring <= 3 && (best < 0 || ring <= 1); ring++) {
      for (let a = -ring; a <= ring; a++) {
        for (let b = -ring; b <= ring; b++) {
          if (Math.max(Math.abs(a), Math.abs(b)) !== ring) continue;
          const cell = this.cells.get(`${cellX + a}:${cellY + b}`);
          if (cell) for (const index of cell) visit(index);
        }
      }
    }
    if (best < 0) for (let index = 0; index < line.count; index++) visit(index);
    return best;
  }

  /** Station, distance along and signed offset of a point, interpolated between stations. */
  place(x: number, y: number): RoadPlace {
    const { line } = this;
    const station = this.nearest(x, y);
    const dx = x - line.x[station]!;
    const dy = y - line.y[station]!;
    return {
      station,
      s: line.s[station]! + dx * line.tx[station]! + dy * line.ty[station]!,
      d: dx * line.rx[station]! + dy * line.ry[station]!,
    };
  }

  /** World position of a road place, between stations. */
  point(s: number, d: number): WorldPoint {
    const { line } = this;
    const { x, y, rx, ry } = line;
    const index = stationAt(line, s);
    const next = Math.min(index + 1, line.count - 1);
    const span = line.s[next]! - line.s[index]!;
    const mix = span > 0 ? (s - line.s[index]!) / span : 0;
    const blend = (values: Float32Array): number =>
      values[index]! + (values[next]! - values[index]!) * mix;
    return [blend(x) + blend(rx) * d, blend(y) + blend(ry) * d];
  }
}

/** Largest station index at or before distance s. */
export function stationAt(line: StreetLine, s: number): number {
  let low = 0;
  let high = line.count - 2;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (line.s[middle]! <= s) low = middle;
    else high = middle - 1;
  }
  return Math.max(low, 0);
}

/** Stable pseudo-random in 0..1 from any two numbers: the same tree grows the same every load. */
export function seeded(a: number, b = 0, salt = 0): number {
  const value = Math.sin(a * 12.9898 + b * 78.233 + salt * 37.719) * 43758.5453;
  return value - Math.floor(value);
}
