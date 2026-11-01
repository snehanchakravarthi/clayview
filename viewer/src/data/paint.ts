/**
 * Turns the noisy polygons the pipeline found on the road into clean paint. A polygon is read in
 * road coordinates, where lane lines run straight, and redrawn as what it most resembles: a stroke
 * along the road, a bar across it, an arrow, a zebra, or a tidied figure such as a zigzag. Smears
 * too thick to be paint are reduced to the line that caused them, or dropped.
 */

import type { RoadMarking, WorldPoint } from "./scene";
import { clamp, gaussian, median, StreetIndex, type StreetLine } from "./street";

const SIMPLIFY_M = 0.14;
const THICK_SMEAR_M = 0.72;
const MAX_LINE_WIDTH = 0.24;
const MIN_LINE_WIDTH = 0.13;
const SMEAR_LINE_WIDTH = 0.15;
const MIN_SMEAR_LENGTH = 7;
const STROKE_RATIO = 2.6;
const BIN_M = 1;
const MIN_BAR_THICKNESS = 0.28;
const MAX_BAR_THICKNESS = 0.55;
const ZEBRA_PITCH = 1;
const ZEBRA_STRIPE = 0.52;
const MIN_ZEBRA_LENGTH = 2.6;
const OFF_ROAD_MARGIN = 1.1;
const MIN_FIGURE_AREA = 0.12;
/** A compact lump this thick isn't a stroke of paint, it's a fragment of something smeared. */
const LUMP_STROKE_M = 0.4;
const LUMP_FILL = 0.48;
/** Zigzags and chevrons are thin. A big or thick figure is a smear caught at a slow turn. */
const MAX_FIGURE_AREA = 7;
const MAX_FIGURE_STROKE = 0.42;
const MIN_FIGURE_REACH = 7;
const ZIGZAG_CORNER_M = 0.32;
const MIN_ZIGZAG_CORNERS = 8;
const STRAIGHT_STROKE_M = 11;

export type PaintKind = "stroke" | "bar" | "arrow" | "zebra" | "figure";

export interface PaintShape {
  kind: PaintKind;
  /** A simple closed ring in world metres, ready to triangulate. */
  ring: WorldPoint[];
}

export type RoadRing = [s: number, d: number][];

export function ringArea(ring: readonly (readonly [number, number])[]): number {
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    const [ax, ay] = ring[index]!;
    const [bx, by] = ring[(index + 1) % ring.length]!;
    sum += ax * by - bx * ay;
  }
  return Math.abs(sum) / 2;
}

export function ringPerimeter(ring: readonly (readonly [number, number])[]): number {
  let sum = 0;
  for (let index = 0; index < ring.length; index++) {
    const [ax, ay] = ring[index]!;
    const [bx, by] = ring[(index + 1) % ring.length]!;
    sum += Math.hypot(bx - ax, by - ay);
  }
  return sum;
}

function pointLineDistance(
  point: readonly [number, number],
  a: readonly [number, number],
  b: readonly [number, number],
): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length = dx * dx + dy * dy;
  if (length === 0) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  const t = clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length, 0, 1);
  return Math.hypot(point[0] - (a[0] + dx * t), point[1] - (a[1] + dy * t));
}

/** Douglas-Peucker on an open polyline. */
export function simplifyLine<T extends readonly [number, number]>(
  points: readonly T[],
  tolerance: number,
): T[] {
  if (points.length < 3) return [...points];
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [from, to] = stack.pop()!;
    let furthest = -1;
    let distance = tolerance;
    for (let index = from + 1; index < to; index++) {
      const here = pointLineDistance(points[index]!, points[from]!, points[to]!);
      if (here > distance) {
        distance = here;
        furthest = index;
      }
    }
    if (furthest < 0) continue;
    keep[furthest] = 1;
    stack.push([from, furthest], [furthest, to]);
  }
  return points.filter((_, index) => keep[index]);
}

/** Douglas-Peucker on a closed ring, split at its two most distant points so it stays closed. */
export function simplifyRing<T extends readonly [number, number]>(
  ring: readonly T[],
  tolerance: number,
): T[] {
  if (ring.length < 5) return [...ring];
  let far = 0;
  let distance = -1;
  for (let index = 1; index < ring.length; index++) {
    const here = Math.hypot(ring[index]![0] - ring[0]![0], ring[index]![1] - ring[0]![1]);
    if (here > distance) {
      distance = here;
      far = index;
    }
  }
  const first = simplifyLine(ring.slice(0, far + 1), tolerance);
  const second = simplifyLine([...ring.slice(far), ring[0]!], tolerance);
  return [...first.slice(0, -1), ...second.slice(0, -1)];
}

/** Lowest and highest d of a ring along the line s = constant. */
function spanAt(ring: RoadRing, s: number): [number, number] | null {
  let low = Infinity;
  let high = -Infinity;
  for (let index = 0; index < ring.length; index++) {
    const [as, ad] = ring[index]!;
    const [bs, bd] = ring[(index + 1) % ring.length]!;
    if ((as <= s && bs > s) || (bs <= s && as > s)) {
      const d = ad + ((bd - ad) * (s - as)) / (bs - as);
      low = Math.min(low, d);
      high = Math.max(high, d);
    }
  }
  return low <= high ? [low, high] : null;
}

function extent(ring: RoadRing): { s0: number; s1: number; d0: number; d1: number } {
  let s0 = Infinity;
  let s1 = -Infinity;
  let d0 = Infinity;
  let d1 = -Infinity;
  for (const [s, d] of ring) {
    s0 = Math.min(s0, s);
    s1 = Math.max(s1, s);
    d0 = Math.min(d0, d);
    d1 = Math.max(d1, d);
  }
  return { s0, s1, d0, d1 };
}

/** A centreline through a shape that runs along the road: d at each step of s. */
function centreline(
  ring: RoadRing,
  s0: number,
  s1: number,
  sigma: number,
): { s: number[]; d: number[] } {
  const steps = Math.max(Math.round((s1 - s0) / BIN_M), 2);
  const stations: number[] = [];
  const values: number[] = [];
  for (let step = 0; step <= steps; step++) {
    const s = s0 + ((s1 - s0) * (step + (step === 0 ? 0.08 : step === steps ? -0.08 : 0))) / steps;
    const span = spanAt(ring, s);
    if (!span) continue;
    stations.push(s);
    values.push((span[0] + span[1]) / 2);
  }
  if (values.length < 2) return { s: stations, d: values };
  if (s1 - s0 < STRAIGHT_STROKE_M) return straightened(stations, values);
  const steady = values.length >= 5 ? median(values, 2) : Float32Array.from(values);
  return { s: stations, d: Array.from(gaussian(steady, sigma)) };
}

/** Paint is laid with a machine: a short stroke is straight in road coordinates. */
function straightened(s: number[], d: number[]): { s: number[]; d: number[] } {
  const count = s.length;
  const meanS = s.reduce((sum, value) => sum + value, 0) / count;
  const meanD = d.reduce((sum, value) => sum + value, 0) / count;
  let across = 0;
  let along = 0;
  for (let index = 0; index < count; index++) {
    across += (s[index]! - meanS) * (d[index]! - meanD);
    along += (s[index]! - meanS) ** 2;
  }
  const slope = along > 0 ? across / along : 0;
  return { s, d: s.map((value) => meanD + slope * (value - meanS)) };
}

/** A ribbon with softened ends, following a centreline, as a closed ring in road coordinates. */
function ribbon(line: { s: number[]; d: number[] }, width: number): RoadRing {
  const half = width / 2;
  const { s, d } = line;
  const last = s.length - 1;
  const trim = half * 0.45;
  const right: RoadRing = [];
  const left: RoadRing = [];
  for (let index = 0; index <= last; index++) {
    right.push([s[index]!, d[index]! + half]);
    left.push([s[index]!, d[index]! - half]);
  }
  return [
    [s[0]! - trim, d[0]! - half * 0.4],
    [s[0]! - trim, d[0]! + half * 0.4],
    ...right,
    [s[last]! + trim, d[last]! + half * 0.4],
    [s[last]! + trim, d[last]! - half * 0.4],
    ...left.reverse(),
  ];
}

/** A bar across the road. */
function bar(s: number, d0: number, d1: number, thickness: number): RoadRing {
  const half = thickness / 2;
  const steps = Math.max(Math.ceil((d1 - d0) / 3), 1);
  const near: RoadRing = [];
  const far: RoadRing = [];
  for (let step = 0; step <= steps; step++) {
    const d = d0 + ((d1 - d0) * step) / steps;
    near.push([s - half, d]);
    far.push([s + half, d]);
  }
  return [...near, ...far.reverse()];
}

/** A straight-ahead lane arrow in road coordinates. `direction` is +1 with ego's travel. */
export function arrowRing(s: number, d: number, length: number, direction: 1 | -1): RoadRing {
  const half = length / 2;
  const headLength = length * 0.36;
  const headHalf = clamp(length * 0.19, 0.42, 0.62);
  const shaft = clamp(length * 0.038, 0.09, 0.14);
  const neck = half - headLength;
  const outline: [number, number][] = [
    [-half, -shaft],
    [neck, -shaft],
    [neck - headLength * 0.1, -headHalf],
    [half, 0],
    [neck - headLength * 0.1, headHalf],
    [neck, shaft],
    [-half, shaft],
  ];
  return outline.map(([along, across]) => [s + along * direction, d + across * direction]);
}

function isArrow(ring: RoadRing, area: number): { direction: 1 | -1 } | null {
  const { s0, s1, d0, d1 } = extent(ring);
  const length = s1 - s0;
  const width = d1 - d0;
  if (length < 2.2 || length > 8.5 || width < 0.45 || width > 2.4) return null;
  if (area < 0.35 || area > 5.5) return null;
  const samples = 16;
  const widths: number[] = [];
  for (let step = 0; step < samples; step++) {
    const span = spanAt(ring, s0 + (length * (step + 0.5)) / samples);
    widths.push(span ? span[1] - span[0] : 0);
  }
  const sorted = [...widths].sort((a, b) => a - b);
  const typical = sorted[samples >> 1]!;
  const widest = sorted[samples - 1]!;
  if (widest < Math.max(typical * 1.5, 0.5)) return null;
  const peak = widths.indexOf(widest) / (samples - 1);
  if (peak > 0.4 && peak < 0.6) return null;
  return { direction: peak >= 0.5 ? 1 : -1 };
}

function zebra(ring: RoadRing): RoadRing[] {
  const { s0, s1, d0, d1 } = extent(ring);
  const middle = (s0 + s1) / 2;
  const half = Math.max(s1 - s0, MIN_ZEBRA_LENGTH) / 2;
  const count = Math.max(Math.round((d1 - d0) / ZEBRA_PITCH), 1);
  const start = (d0 + d1) / 2 - ((count - 1) * ZEBRA_PITCH) / 2;
  const stripes: RoadRing[] = [];
  for (let index = 0; index < count; index++) {
    const d = start + index * ZEBRA_PITCH;
    const edge = ZEBRA_STRIPE / 2;
    stripes.push([
      [middle - half, d - edge],
      [middle - half, d + edge],
      [middle + half, d + edge],
      [middle + half, d - edge],
    ]);
  }
  return stripes;
}

/** Principal axis of a ring, to catch straight strokes running at an angle to the road. */
function principal(ring: RoadRing): { length: number; width: number; angle: number } {
  let meanS = 0;
  let meanD = 0;
  for (const [s, d] of ring) {
    meanS += s;
    meanD += d;
  }
  meanS /= ring.length;
  meanD /= ring.length;
  let ss = 0;
  let dd = 0;
  let sd = 0;
  for (const [s, d] of ring) {
    ss += (s - meanS) ** 2;
    dd += (d - meanD) ** 2;
    sd += (s - meanS) * (d - meanD);
  }
  const angle = 0.5 * Math.atan2(2 * sd, ss - dd);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  let lowA = Infinity;
  let highA = -Infinity;
  let lowB = Infinity;
  let highB = -Infinity;
  for (const [s, d] of ring) {
    const a = (s - meanS) * cos + (d - meanD) * sin;
    const b = -(s - meanS) * sin + (d - meanD) * cos;
    lowA = Math.min(lowA, a);
    highA = Math.max(highA, a);
    lowB = Math.min(lowB, b);
    highB = Math.max(highB, b);
  }
  return { length: highA - lowA, width: highB - lowB, angle };
}

/** A straight bar along a principal axis, for strokes that cross the road at an angle. */
function slantedBar(
  axis: { length: number; angle: number },
  s: number,
  d: number,
  width: number,
): RoadRing {
  const cos = Math.cos(axis.angle);
  const sin = Math.sin(axis.angle);
  const half = axis.length / 2;
  const edge = width / 2;
  return [
    [s - half * cos + edge * sin, d - half * sin - edge * cos],
    [s + half * cos + edge * sin, d + half * sin - edge * cos],
    [s + half * cos - edge * sin, d + half * sin + edge * cos],
    [s - half * cos - edge * sin, d - half * sin + edge * cos],
  ];
}

/** Read one marking and redraw it. Returns nothing when it can't be believed as paint. */
export function cleanMarking(marking: RoadMarking, index: StreetIndex): PaintShape[] {
  const placed: RoadRing = marking.points.map(([x, y]) => {
    const place = index.place(x, y);
    return [place.s, place.d];
  });
  if (placed.length < 3) return [];
  const ring = simplifyRing(placed, SIMPLIFY_M);
  if (ring.length < 3) return [];

  const toWorld = (road: RoadRing): WorldPoint[] => road.map(([s, d]) => index.point(s, d));
  const shape = (kind: PaintKind, road: RoadRing): PaintShape => ({ kind, ring: toWorld(road) });

  if (!onTheRoad(placed, index.line)) return [];
  const area = ringArea(ring);
  const stroke = (2 * area) / ringPerimeter(ring);
  const { s0, s1, d0, d1 } = extent(ring);
  const length = s1 - s0;
  const width = d1 - d0;

  if (marking.kind === "crosswalk") return zebra(ring).map((stripe) => shape("zebra", stripe));

  if (stroke > THICK_SMEAR_M) {
    if (length < MIN_SMEAR_LENGTH || length < width * 1.6) return [];
    return [shape("stroke", ribbon(centreline(ring, s0, s1, 6), SMEAR_LINE_WIDTH))];
  }

  const arrow = isArrow(ring, area);
  if (arrow) {
    return [
      shape(
        "arrow",
        arrowRing((s0 + s1) / 2, (d0 + d1) / 2, clamp(length, 3.4, 5.6), arrow.direction),
      ),
    ];
  }

  if (width > length * STROKE_RATIO * 0.6 && length < 1.2) {
    const thickness = clamp(length, MIN_BAR_THICKNESS, MAX_BAR_THICKNESS);
    return [shape("bar", bar((s0 + s1) / 2, d0, d1, thickness))];
  }

  // A line can drift across the road over a long way. Weaving this much in a short way is a zigzag.
  const steady = width < Math.max(1, length * 0.15);
  if (length >= width * STROKE_RATIO && length > 0.7 && steady) {
    const lineWidth = clamp(stroke, MIN_LINE_WIDTH, MAX_LINE_WIDTH);
    return [shape("stroke", ribbon(centreline(ring, s0, s1, 3), lineWidth))];
  }

  const axis = principal(ring);
  const straight = axis.width < Math.max(stroke * 2.2, 0.6);
  if (axis.length >= axis.width * STROKE_RATIO && axis.length > 1 && straight) {
    const lineWidth = clamp(stroke, MIN_LINE_WIDTH, MAX_LINE_WIDTH);
    return [shape("stroke", slantedBar(axis, (s0 + s1) / 2, (d0 + d1) / 2, lineWidth))];
  }

  const lump = stroke > LUMP_STROKE_M && area > length * width * LUMP_FILL;
  const believable = area >= MIN_FIGURE_AREA && area <= MAX_FIGURE_AREA && stroke <= MAX_FIGURE_STROKE;
  if (!believable || lump) return [];
  // A real zigzag keeps its corners when looked at coarsely. Stairs off the pixel grid don't, and a
  // scrap that small tidies up into the bar it nearly is, or goes.
  const coarse = simplifyRing(ring, ZIGZAG_CORNER_M);
  if (ringPerimeter(ring) >= MIN_FIGURE_REACH && coarse.length >= MIN_ZIGZAG_CORNERS) {
    return [shape("figure", coarse)];
  }
  if (axis.length < axis.width * 1.6) return [];
  const lineWidth = clamp(stroke, MIN_LINE_WIDTH, MAX_LINE_WIDTH);
  return [shape("bar", slantedBar(axis, (s0 + s1) / 2, (d0 + d1) / 2, lineWidth))];
}

/** True when most of a shape sits between the road's edges. */
function onTheRoad(ring: RoadRing, line: StreetLine): boolean {
  let inside = 0;
  for (const [s, d] of ring) {
    const station = nearestStation(line, s);
    if (d > line.left[station]! - OFF_ROAD_MARGIN && d < line.right[station]! + OFF_ROAD_MARGIN) {
      inside++;
    }
  }
  return inside * 2 >= ring.length;
}

function nearestStation(line: StreetLine, s: number): number {
  const first = line.s[0]!;
  const span = line.s[line.count - 1]! - first;
  const guess = Math.round(((s - first) / span) * (line.count - 1));
  return clamp(guess, 0, line.count - 1);
}

export function cleanMarkings(markings: readonly RoadMarking[], index: StreetIndex): PaintShape[] {
  return markings.flatMap((marking) => cleanMarking(marking, index));
}
