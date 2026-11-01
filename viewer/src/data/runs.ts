/**
 * Walls, fences and building fronts arrive as polylines of noisy offsets from the road. This reads
 * them in road coordinates, breaks them where they jump, settles the noise and keeps them behind
 * the road's edge, so what gets built follows the street instead of zigzagging across it.
 */

import type { WorldPoint } from "./scene";
import { gaussian, median, stationAt, type StreetIndex, type StreetLine } from "./street";

export interface RunOptions {
  /** A jump in offset bigger than this starts a new run. */
  splitJump: number;
  /** A gap along the road bigger than this starts a new run. */
  splitGap: number;
  /** Shorter runs are noise. */
  minLength: number;
  /** Keep this far beyond the road's edge. */
  clearance: number;
  /** Also stay behind the sidewalk and verge the road line knows about. */
  behindVerge: boolean;
  /** Spacing of the points in the result. */
  step: number;
}

export interface CleanRun {
  points: WorldPoint[];
  /** The same points in road coordinates. */
  s: number[];
  d: number[];
  length: number;
}

/** How far out from the centre a thing on this side has to stay, at a place on the road. */
export function verge(
  line: StreetLine,
  s: number,
  sign: number,
  clearance: number,
  behindVerge: boolean,
): number {
  const index = stationAt(line, s);
  const edge = Math.abs(sign > 0 ? line.right[index]! : line.left[index]!);
  const curb = sign > 0 ? line.curbRight[index]! : line.curbLeft[index]!;
  const walk = sign > 0 ? line.walkRight[index]! : line.walkLeft[index]!;
  return edge + clearance + (behindVerge ? curb * 0.24 + walk : 0);
}

function finish(
  s: number[],
  d: number[],
  sign: number,
  index: StreetIndex,
  options: RunOptions,
): CleanRun | null {
  const length = s[s.length - 1]! - s[0]!;
  if (length < options.minLength) return null;
  const steady = d.length >= 3 ? median(d, 1) : Float32Array.from(d);
  const smooth = gaussian(steady, 1);
  const steps = Math.max(Math.round(length / options.step), 1);
  const outS: number[] = [];
  const outD: number[] = [];
  const points: WorldPoint[] = [];
  let source = 0;
  for (let step = 0; step <= steps; step++) {
    const at = s[0]! + (length * step) / steps;
    while (source + 2 < s.length && s[source + 1]! < at) source++;
    const span = s[source + 1]! - s[source]!;
    const mix = span > 0 ? Math.min((at - s[source]!) / span, 1) : 0;
    const offset = smooth[source]! + (smooth[source + 1]! - smooth[source]!) * mix;
    const edge = verge(index.line, at, sign, options.clearance, options.behindVerge);
    const held = sign > 0 ? Math.max(offset, edge) : Math.min(offset, -edge);
    outS.push(at);
    outD.push(held);
    points.push(index.point(at, held));
  }
  return { points, s: outS, d: outD, length };
}

/** Clean one polyline. It can come back as several runs, or none. */
export function cleanRun(
  points: readonly WorldPoint[],
  side: "left" | "right",
  index: StreetIndex,
  options: RunOptions,
): CleanRun[] {
  const sign = side === "right" ? 1 : -1;
  const placed = points
    .map(([x, y]) => index.place(x, y))
    .filter((place) => place.d * sign > 0)
    .sort((a, b) => a.s - b.s);
  const runs: CleanRun[] = [];
  let s: number[] = [];
  let d: number[] = [];
  const flush = (): void => {
    if (s.length >= 2) {
      const run = finish(s, d, sign, index, options);
      if (run) runs.push(run);
    }
    s = [];
    d = [];
  };
  for (const place of placed) {
    const last = s.length - 1;
    if (last >= 0) {
      if (place.s - s[last]! < 0.05) continue;
      const jump = Math.abs(place.d - d[last]!);
      if (jump > options.splitJump || place.s - s[last]! > options.splitGap) flush();
    }
    s.push(place.s);
    d.push(place.d);
  }
  flush();
  return runs;
}
