import { Vector3, type Color } from "three";
import { cleanRun, verge, type CleanRun, type RunOptions } from "../data/runs";
import type { BuildingRun, WorldPoint } from "../data/scene";
import { clamp, seeded, type StreetIndex } from "../data/street";
import { PALETTE } from "./palette";
import type { Soup } from "./soup";

const FRONT_RULES: RunOptions = {
  splitJump: 7,
  splitGap: 22,
  minLength: 5,
  clearance: 1.4,
  behindVerge: true,
  step: 3,
};

const FLOOR_M = 3.3;
const GROUND_FLOOR_M = 4.2;
const MIN_LOT = 13;
const MAX_LOT = 30;
const MIN_LOT_WIDTH = 8;
const MIN_HEIGHT = 7.5;
const MAX_HEIGHT = 44;
const MIN_DEPTH = 11;
const MAX_DEPTH = 22;
/** Any deeper and the fins throw a comb-shaped shadow right across the road. */
const FIN_DEPTH = 0.13;
const ROAD_GAP = 1.5;
const DEPTH_STEP = 2;
const BACK_SAMPLES = 5;
const MIN_FITTED_DEPTH = 6;
const SAMPLE = new Vector3();

const UP = new Vector3(0, 1, 0);
const P0 = new Vector3();
const P1 = new Vector3();
const P2 = new Vector3();
const P3 = new Vector3();

type Facing = "front" | "plain";

interface Style {
  bay: number;
  /** Window share of the bay, across and up. */
  wide: number;
  tall: number;
  depth: number;
  ribbon: boolean;
  fins: boolean;
}

const STYLES: readonly Style[] = [
  { bay: 3.3, wide: 0.52, tall: 0.56, depth: 0.2, ribbon: false, fins: false },
  { bay: 2.7, wide: 0.42, tall: 0.72, depth: 0.26, ribbon: false, fins: false },
  { bay: 6.5, wide: 0.92, tall: 0.48, depth: 0.32, ribbon: true, fins: false },
  { bay: 1.8, wide: 0.74, tall: 0.8, depth: 0.1, ribbon: false, fins: true },
];

/** A patch of wall, seen from outside: `left` and `right` as the viewer stands facing it. */
function patch(soup: Soup, left: Vector3, right: Vector3, y0: number, y1: number, color: Color, out = 0, normal?: Vector3): void {
  P0.copy(left).setY(y0);
  P1.copy(right).setY(y0);
  P2.copy(right).setY(y1);
  P3.copy(left).setY(y1);
  if (out && normal) {
    P0.addScaledVector(normal, out);
    P1.addScaledVector(normal, out);
    P2.addScaledVector(normal, out);
    P3.addScaledVector(normal, out);
  }
  soup.quad(P0, P1, P2, P3, color);
}

const A = new Vector3();
const B = new Vector3();
const C = new Vector3();
const D = new Vector3();
const INNER_A = new Vector3();
const INNER_B = new Vector3();
const INNER_C = new Vector3();
const INNER_D = new Vector3();

/** A window: a recessed pane with its four reveals. */
function opening(soup: Soup, left: Vector3, right: Vector3, y0: number, y1: number, depth: number, normal: Vector3, glass: Color, reveal: Color): void {
  A.copy(left).setY(y0);
  B.copy(right).setY(y0);
  C.copy(right).setY(y1);
  D.copy(left).setY(y1);
  INNER_A.copy(A).addScaledVector(normal, -depth);
  INNER_B.copy(B).addScaledVector(normal, -depth);
  INNER_C.copy(C).addScaledVector(normal, -depth);
  INNER_D.copy(D).addScaledVector(normal, -depth);
  soup.quad(INNER_A, INNER_B, INNER_C, INNER_D, glass);
  soup.quad(A, INNER_A, INNER_D, D, reveal);
  soup.quad(B, C, INNER_C, INNER_B, reveal);
  soup.quad(INNER_A, A, B, INNER_B, reveal);
  soup.quad(D, INNER_D, INNER_C, C, PALETTE.massingShade);
}

const ALONG = new Vector3();
const L = new Vector3();
const R = new Vector3();
const NORMAL = new Vector3();

/** One whole face of a block, floor by floor. */
function facade(soup: Soup, left: Vector3, right: Vector3, y0: number, y1: number, style: Style, wall: Color, seed: number): void {
  ALONG.subVectors(right, left);
  const width = ALONG.length();
  ALONG.normalize();
  NORMAL.crossVectors(ALONG, UP).normalize();
  const bays = Math.max(Math.round(width / style.bay), 1);
  const bay = width / bays;
  const floors = Math.max(Math.round((y1 - y0) / FLOOR_M), 1);
  const floor = (y1 - y0) / floors;
  const sill = floor * (1 - style.tall) * 0.46;
  const head = floor * (1 - style.tall) * 0.54;
  const windowWidth = bay * style.wide;
  const pier = (bay - windowWidth) / 2;

  for (let level = 0; level < floors; level++) {
    const base = y0 + level * floor;
    patch(soup, left, right, base, base + sill, wall);
    patch(soup, left, right, base + floor - head, base + floor, wall);
    const low = base + sill;
    const high = base + floor - head;
    if (style.ribbon) {
      L.copy(left).addScaledVector(ALONG, pier);
      R.copy(right).addScaledVector(ALONG, -pier);
      patch(soup, left, L, low, high, wall);
      patch(soup, R, right, low, high, wall);
      opening(soup, L, R, low, high, style.depth, NORMAL, PALETTE.massingGlass, PALETTE.massingShade);
      continue;
    }
    for (let column = 0; column < bays; column++) {
      const start = column * bay;
      L.copy(left).addScaledVector(ALONG, start);
      R.copy(left).addScaledVector(ALONG, start + pier);
      patch(soup, L, R, low, high, wall);
      L.copy(left).addScaledVector(ALONG, start + pier);
      R.copy(left).addScaledVector(ALONG, start + pier + windowWidth);
      const lit = seeded(seed, level * 13 + column) > 0.1;
      opening(soup, L, R, low, high, style.depth, NORMAL, lit ? PALETTE.massingGlass : PALETTE.massingShade, PALETTE.massingShade);
      L.copy(left).addScaledVector(ALONG, start + pier + windowWidth);
      R.copy(left).addScaledVector(ALONG, start + bay);
      patch(soup, L, R, low, high, wall);
    }
  }
  if (style.fins) {
    for (let column = 0; column <= bays; column++) {
      L.copy(left).addScaledVector(ALONG, column * bay - 0.05);
      R.copy(left).addScaledVector(ALONG, column * bay + 0.05);
      patch(soup, L, R, y0, y1, wall, FIN_DEPTH, NORMAL);
      L.copy(left).addScaledVector(ALONG, column * bay + 0.05);
      soup.quad(
        A.copy(L).setY(y0),
        B.copy(L).setY(y0).addScaledVector(NORMAL, FIN_DEPTH),
        C.copy(L).setY(y1).addScaledVector(NORMAL, FIN_DEPTH),
        D.copy(L).setY(y1),
        PALETTE.massingShade,
      );
    }
  }
}

/** Glazed ground floor: columns with a deep recess and a thin canopy over the pavement. */
function groundFloor(soup: Soup, left: Vector3, right: Vector3, height: number, wall: Color): void {
  ALONG.subVectors(right, left);
  const width = ALONG.length();
  ALONG.normalize();
  NORMAL.crossVectors(ALONG, UP).normalize();
  const bays = Math.max(Math.round(width / 5.2), 1);
  const bay = width / bays;
  const column = 0.5;
  for (let index = 0; index < bays; index++) {
    L.copy(left).addScaledVector(ALONG, index * bay);
    R.copy(left).addScaledVector(ALONG, index * bay + column);
    patch(soup, L, R, 0, height, wall);
    L.copy(left).addScaledVector(ALONG, index * bay + column);
    R.copy(left).addScaledVector(ALONG, (index + 1) * bay);
    opening(soup, L, R, 0.45, height - 0.45, 0.42, NORMAL, PALETTE.massingGlass, PALETTE.massingShade);
    patch(soup, L, R, 0, 0.45, wall);
    patch(soup, L, R, height - 0.45, height, wall);
  }
}

interface Block {
  frontLeft: Vector3;
  frontRight: Vector3;
  backLeft: Vector3;
  backRight: Vector3;
}

function blockAt(frontLeft: Vector3, frontRight: Vector3, depth: number): Block {
  const along = new Vector3().subVectors(frontRight, frontLeft).normalize();
  const out = new Vector3().crossVectors(along, UP).normalize();
  return {
    frontLeft,
    frontRight,
    backLeft: frontLeft.clone().addScaledVector(out, -depth),
    backRight: frontRight.clone().addScaledVector(out, -depth),
  };
}

function inset(block: Block, by: number): Block {
  const across = new Vector3().subVectors(block.frontRight, block.frontLeft).normalize();
  const back = new Vector3().subVectors(block.backLeft, block.frontLeft).normalize();
  const move = (point: Vector3, a: number, b: number): Vector3 =>
    point.clone().addScaledVector(across, a * by).addScaledVector(back, b * by);
  return {
    frontLeft: move(block.frontLeft, 1, 1),
    frontRight: move(block.frontRight, -1, 1),
    backLeft: move(block.backLeft, 1, -1),
    backRight: move(block.backRight, -1, -1),
  };
}

function roof(soup: Soup, block: Block, y: number, color: Color): void {
  soup.quad(
    A.copy(block.frontLeft).setY(y),
    B.copy(block.frontRight).setY(y),
    C.copy(block.backRight).setY(y),
    D.copy(block.backLeft).setY(y),
    color,
  );
}

function parapet(soup: Soup, block: Block, y: number, wall: Color): void {
  const rim = 0.32;
  const inner = inset(block, rim);
  roof(soup, inner, y, PALETTE.massingShade);
  const top = y + 0.5;
  const corners: [Vector3, Vector3, Vector3, Vector3][] = [
    [block.frontLeft, block.frontRight, inner.frontLeft, inner.frontRight],
    [block.frontRight, block.backRight, inner.frontRight, inner.backRight],
    [block.backRight, block.backLeft, inner.backRight, inner.backLeft],
    [block.backLeft, block.frontLeft, inner.backLeft, inner.frontLeft],
  ];
  for (const [outerA, outerB, innerA, innerB] of corners) {
    patch(soup, outerA, outerB, y, top, wall);
    patch(soup, innerB, innerA, y, top, PALETTE.massingShade);
    soup.quad(
      A.copy(outerA).setY(top),
      B.copy(outerB).setY(top),
      C.copy(innerB).setY(top),
      D.copy(innerA).setY(top),
      wall,
    );
  }
}

function plainBlock(soup: Soup, block: Block, y0: number, y1: number, wall: Color, style: Style, seed: number, facing: Facing): void {
  const { frontLeft, frontRight, backLeft, backRight } = block;
  if (facing === "front") {
    facade(soup, frontLeft, frontRight, y0, y1, style, wall, seed);
    facade(soup, frontRight, backRight, y0, y1, style, wall, seed + 1);
    facade(soup, backLeft, frontLeft, y0, y1, style, wall, seed + 2);
  } else {
    patch(soup, frontLeft, frontRight, y0, y1, wall);
    patch(soup, frontRight, backRight, y0, y1, wall);
    patch(soup, backLeft, frontLeft, y0, y1, wall);
  }
  patch(soup, backRight, backLeft, y0, y1, wall);
}

function building(soup: Soup, block: Block, height: number, seed: number): void {
  const wall = (seeded(seed, 31) > 0.5 ? PALETTE.massing : PALETTE.massingShade).clone();
  const style = STYLES[Math.floor(seeded(seed, 32) * STYLES.length)]!;
  const shop = seeded(seed, 33) > 0.45;
  const base = shop ? GROUND_FLOOR_M : 0;
  if (shop) {
    groundFloor(soup, block.frontLeft, block.frontRight, base, wall);
    patch(soup, block.frontRight, block.backRight, 0, base, wall);
    patch(soup, block.backLeft, block.frontLeft, 0, base, wall);
    patch(soup, block.backRight, block.backLeft, 0, base, wall);
  }
  const steps = height > 20 && seeded(seed, 34) > 0.45;
  const podium = steps ? base + FLOOR_M * (2 + Math.floor(seeded(seed, 35) * 3)) : height;
  plainBlock(soup, block, base, podium, wall, style, seed, "front");
  if (steps) {
    roof(soup, block, podium, PALETTE.massingShade);
    const tower = inset(block, 2.4);
    plainBlock(soup, tower, podium, height, wall, style, seed + 3, "front");
    parapet(soup, tower, height, wall);
  } else {
    parapet(soup, block, height, wall);
  }
  if (seeded(seed, 36) > 0.5) {
    const top = steps ? inset(block, 2.4) : block;
    const plant = inset(top, Math.min(4, shortestSide(top) * 0.3));
    const y = height + 0.02;
    plainBlock(soup, plant, y, y + 2.4, PALETTE.massingShade, style, seed, "plain");
    roof(soup, plant, y + 2.4, PALETTE.massingShade);
  }
}

function shortestSide(block: Block): number {
  return Math.min(
    block.frontLeft.distanceTo(block.frontRight),
    block.frontLeft.distanceTo(block.backLeft),
  );
}

function worldPoint(point: WorldPoint): Vector3 {
  return new Vector3(point[0], 0, -point[1]);
}

/** True when a point of the shot's world, in three.js axes, is clear of every stretch of road. */
function clearOfRoad(point: Vector3, index: StreetIndex): boolean {
  const place = index.place(point.x, -point.z);
  const sign = place.d >= 0 ? 1 : -1;
  return Math.abs(place.d) >= verge(index.line, place.s, sign, ROAD_GAP, false);
}

/**
 * A lot's straight front is a chord, and on the outside of a bend a chord swings in towards the
 * road. How far it has to move out to stay behind the verge.
 */
function chordIntrusion(index: StreetIndex, from: number, to: number, front: number, side: number): number {
  const a = index.point(from, front);
  const b = index.point(to, front);
  const place = index.place((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
  const needed = verge(index.line, place.s, side, FRONT_RULES.clearance, FRONT_RULES.behindVerge);
  return Math.max(needed - place.d * side, 0);
}

/**
 * Make a lot as deep as it can go. Where the road doubles back, a block raised from one stretch can
 * reach the next, so it gets shallower, and gives up rather than stand on the road.
 */
function fittedBlock(frontLeft: Vector3, frontRight: Vector3, depth: number, index: StreetIndex): Block | null {
  for (let fitted = depth; fitted >= MIN_FITTED_DEPTH; fitted -= DEPTH_STEP) {
    const block = blockAt(frontLeft, frontRight, fitted);
    let clear = true;
    for (let step = 0; step <= BACK_SAMPLES && clear; step++) {
      const mix = step / BACK_SAMPLES;
      SAMPLE.lerpVectors(block.backLeft, block.backRight, mix);
      clear = clearOfRoad(SAMPLE, index);
      SAMPLE.lerpVectors(block.frontLeft, block.backLeft, mix);
      clear &&= clearOfRoad(SAMPLE, index);
      SAMPLE.lerpVectors(block.frontRight, block.backRight, mix);
      clear &&= clearOfRoad(SAMPLE, index);
    }
    if (clear) return block;
  }
  return null;
}

/** Split a cleaned building front into lots and raise each one. */
function raiseRun(soup: Soup, run: CleanRun, hint: number, side: 1 | -1, index: StreetIndex): number {
  const seed = run.s[0]! * 0.37 + side;
  const base = clamp(hint * 1.25, MIN_HEIGHT + 2, MAX_HEIGHT * 0.7);
  let lots = 0;
  // A building only glimpsed leaves a short run. It's still a whole building, not a sliver of one.
  const first = run.s[0]!;
  const last = run.s[run.s.length - 1]!;
  const widen = Math.max((MIN_LOT - (last - first)) / 2, 0);
  let from = first - widen;
  const end = last + widen;
  while (end - from > 4) {
    const lotSeed = seed + lots * 7.3;
    const wanted = MIN_LOT + seeded(lotSeed, 41) * (MAX_LOT - MIN_LOT);
    // A sliver left over at the end of a run would come out as an absurdly thin tower.
    const to = end - (from + wanted) < MIN_LOT * 0.6 ? end : from + wanted;
    if (to - from < MIN_LOT_WIDTH) break;
    const dAt = (s: number): number => {
      let source = 0;
      while (source + 2 < run.s.length && run.s[source + 1]! < s) source++;
      const span = run.s[source + 1]! - run.s[source]!;
      const mix = span > 0 ? clamp((s - run.s[source]!) / span, 0, 1) : 0;
      return run.d[source]! + (run.d[source + 1]! - run.d[source]!) * mix;
    };
    const setback = seeded(lotSeed, 42) * 1.6;
    let front = (dAt(from) + dAt(to)) / 2 + side * setback;
    front += side * chordIntrusion(index, from, to, front, side);
    const a = worldPoint(index.point(from, front));
    const b = worldPoint(index.point(to, front));
    // Facing a building from the road, forward along the road is on your left on the right side.
    const [frontLeft, frontRight] = side > 0 ? [b, a] : [a, b];
    const wantedDepth = MIN_DEPTH + seeded(lotSeed, 43) * (MAX_DEPTH - MIN_DEPTH);
    const block = fittedBlock(frontLeft, frontRight, wantedDepth, index);
    if (block) {
      const rawHeight = base * (0.7 + seeded(lotSeed, 44) * 0.85);
      const floors = Math.max(Math.round((rawHeight - GROUND_FLOOR_M) / FLOOR_M), 1);
      const height = clamp(GROUND_FLOOR_M + floors * FLOOR_M, MIN_HEIGHT, MAX_HEIGHT);
      soup.at(a.x, a.z);
      building(soup, block, height, lotSeed);
      lots++;
    }
    from = to + 0.8 + seeded(lotSeed, 45) * 2.4;
  }
  return lots;
}

/** Raise the street's buildings. Returns how many separate blocks went up. */
export function layMassing(soup: Soup, runs: readonly BuildingRun[], index: StreetIndex): number {
  let lots = 0;
  for (const run of runs) {
    const side = run.side === "right" ? 1 : -1;
    for (const clean of cleanRun(run.points, run.side, index, FRONT_RULES)) {
      lots += raiseRun(soup, clean, run.height, side, index);
    }
  }
  return lots;
}
