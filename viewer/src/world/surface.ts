import {
  type BufferGeometry,
  Color,
  CylinderGeometry,
  LatheGeometry,
  Matrix4,
  Quaternion,
  ShapeUtils,
  Vector2,
  Vector3,
} from "three";
import type { PaintShape } from "../data/paint";
import type { StreetObject } from "../data/scene";
import { clamp, seeded, smoothstep, type StreetIndex, type StreetLine } from "../data/street";
import { PALETTE } from "./palette";
import type { Soup } from "./soup";

const CURB_HEIGHT = 0.15;
const CURB_WIDTH = 0.24;
const WALK_RISE = 0.11;
const BARE_WALK_RISE = 0.05;
const GRASS_RISE = 0.06;
const BARE_GRASS_RISE = 0.03;
const SHOULDER = 0.9;
const GROUND = -0.03;
const PAINT_HEIGHT = 0.014;
const JOINT_HEIGHT = 0.004;
const JOINT_SPACING = 2.4;
const JOINT_WIDTH = 0.035;
const ISLAND_MARGIN = 0.25;
const ISLAND_RADIUS = 0.85;
const ISLAND_LENGTH = 2;
const ISLAND_RISE = 0.14;

const ROAD_COLUMNS = [0, 0.018, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.982, 1] as const;

/** A point on a cross-section, with the surface facing it and its colour. */
interface Point {
  d: number;
  h: number;
  /** Normal in the cross-section plane: along the side's outward direction, and up. */
  out: number;
  up: number;
  color: Color;
}

function point(): Point {
  return { d: 0, h: 0, out: 0, up: 1, color: new Color() };
}

/** Smooth blotches over the road, so a long straight isn't one flat colour. */
function blotch(s: number, d: number): number {
  const cellS = Math.floor(s / 26);
  const cellD = Math.floor((d + 40) / 7);
  const mixS = s / 26 - cellS;
  const mixD = (d + 40) / 7 - cellD;
  const fadeS = mixS * mixS * (3 - 2 * mixS);
  const fadeD = mixD * mixD * (3 - 2 * mixD);
  const at = (a: number, b: number): number => seeded(cellS + a, cellD + b, 3);
  const top = at(0, 0) + (at(1, 0) - at(0, 0)) * fadeS;
  const bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * fadeS;
  return top + (bottom - top) * fadeD;
}

const PAVED = new Color();
const LIP = new Color();
const LAWN = new Color();
const TAIL = new Color();

function roadTone(s: number, d: number, out: Color): Color {
  const value = blotch(s, d);
  out.copy(PALETTE.asphaltDark).lerp(PALETTE.asphaltLight, smoothstep(0.1, 0.9, value));
  return out;
}

function toWorld(line: StreetLine, index: number, d: number, into: Vector2): Vector2 {
  // Three.js axes: X = x, Z = -y. The vector stores X and Z.
  return into.set(line.x[index]! + line.rx[index]! * d, -(line.y[index]! + line.ry[index]! * d));
}

const FROM = new Vector2();
const TO = new Vector2();
const LEFT_A = new Vector2();
const LEFT_B = new Vector2();

/** Lay a strip between one cross-section's points and the next one's. */
function band(
  soup: Soup,
  line: StreetLine,
  index: number,
  sign: number,
  a0: Point,
  b0: Point,
  a1: Point,
  b1: Point,
): void {
  const next = index + 1;
  toWorld(line, index, a0.d, FROM);
  toWorld(line, index, b0.d, TO);
  toWorld(line, next, a1.d, LEFT_A);
  toWorld(line, next, b1.d, LEFT_B);
  const rx0 = line.rx[index]! * sign;
  const rz0 = -line.ry[index]! * sign;
  const rx1 = line.rx[next]! * sign;
  const rz1 = -line.ry[next]! * sign;
  // Outward across the section, forward along the road: the left side runs the other way round.
  const corners: [Vector2, Point, number, number][] = [
    [FROM, a0, rx0, rz0],
    [TO, b0, rx0, rz0],
    [LEFT_B, b1, rx1, rz1],
    [LEFT_A, a1, rx1, rz1],
  ];
  const order = sign > 0 ? [0, 1, 2, 0, 2, 3] : [0, 3, 2, 0, 2, 1];
  for (const at of order) {
    const [place, section, nx, nz] = corners[at]!;
    soup.vertex(
      place.x,
      section.h,
      place.y,
      nx * section.out,
      section.up,
      nz * section.out,
      section.color,
    );
  }
  soup.countTriangles(2);
}

interface Side {
  sign: 1 | -1;
  edge: Float32Array;
  curb: Float32Array;
  walk: Float32Array;
  grass: Float32Array;
}

function sideOf(line: StreetLine, sign: 1 | -1): Side {
  return sign > 0
    ? { sign, edge: line.right, curb: line.curbRight, walk: line.walkRight, grass: line.grassRight }
    : { sign, edge: line.left, curb: line.curbLeft, walk: line.walkLeft, grass: line.grassLeft };
}

const SECTION_POINTS = 17;

interface Rises {
  verge: number;
  grass: number;
  hasWalk: number;
  hasGrass: number;
}

/** How high a side's verge stands at a station. */
function rises(side: Side, index: number): Rises {
  const curb = side.curb[index]!;
  const hasWalk = smoothstep(0, 0.5, side.walk[index]!);
  const hasGrass = smoothstep(0, 0.5, side.grass[index]!);
  const walkRise = hasWalk * (BARE_WALK_RISE + (WALK_RISE - BARE_WALK_RISE) * curb);
  return {
    verge: Math.max(CURB_HEIGHT * curb, walkRise),
    grass: (BARE_GRASS_RISE + GRASS_RISE * Math.max(curb, hasWalk)) * hasGrass,
    hasWalk,
    hasGrass,
  };
}

/**
 * Fill one side's cross-section at a station. Every station has the same points, so strips join up;
 * a feature that isn't there just has zero width. Curbs rise and fall with how sure the data is.
 */
function section(line: StreetLine, side: Side, index: number, points: Point[]): void {
  const { sign } = side;
  const edge = side.edge[index]!;
  const curb = side.curb[index]!;
  const walk = side.walk[index]!;
  const grass = side.grass[index]!;
  const presence = line.presence[index]!;
  const s = line.s[index]!;
  const { verge, grass: grassRise, hasWalk, hasGrass } = rises(side, index);

  const set = (at: number, offset: number, h: number, out: number, up: number, color: Color): void => {
    const target = points[at]!;
    target.d = edge + sign * offset;
    target.h = h;
    const length = Math.hypot(out, up) || 1;
    target.out = out / length;
    target.up = up / length;
    target.color.copy(color).lerp(PALETTE.ground, 1 - presence);
  };

  const paved = PAVED.copy(PALETTE.sidewalk);
  const lip = LIP.copy(PALETTE.sidewalk).lerp(PALETTE.curb, curb);

  // Curb: a small rounded nose facing the road, then a flat top.
  let run = 0;
  set(0, run, 0, -0.9, 0.45, PALETTE.gutter);
  set(1, run + 0.012, verge * 0.72, -0.9, 0.45, lip);
  set(2, run + 0.012, verge * 0.72, -0.8, 0.6, lip);
  set(3, run + 0.045, verge * 0.96, -0.35, 0.94, lip);
  set(4, run + 0.045, verge * 0.96, -0.35, 0.94, lip);
  set(5, run + 0.09, verge, 0, 1, lip);
  run += Math.max(CURB_WIDTH * curb, 0.09);
  set(6, run, verge, 0, 1, lip);

  // Sidewalk.
  set(7, run, verge, 0, 1, paved);
  run += walk;
  set(8, run, verge, 0, 1, paved);

  // A soft step from the paving down to the verge.
  set(9, run, verge, 0.75, 0.66, paved);
  run += hasWalk * 0.05;
  set(10, run, grassRise, 0.75, 0.66, paved);

  // Grass, with a slight mound and a tone that wanders along the road.
  const lawn = LAWN.copy(PALETTE.grass).lerp(PALETTE.grassDeep, blotch(s * 2.2, sign * 3) * 0.8);
  set(11, run, grassRise, 0, 1, lawn);
  run += grass * 0.5;
  set(12, run, grassRise * 1.5, 0, 1, lawn);
  run += grass * 0.5;
  set(13, run, grassRise * 1.1, 0.12, 1, lawn);

  // Whatever the verge ends in, it eases down into the ground beyond.
  const bare = 1 - Math.max(hasWalk, hasGrass);
  const tail = TAIL.copy(hasGrass > 0.5 ? lawn : paved).lerp(PALETTE.shoulder, bare);
  set(14, run, grassRise * 1.1, 0.3, 1, tail);
  run += hasGrass > 0.5 || bare > 0.5 ? 0.5 : 0.06;
  set(15, run, GROUND * 0.5, 0.3, 1, tail.lerp(PALETTE.ground, 0.5));
  run += SHOULDER * 0.5;
  set(16, run, GROUND, 0.1, 1, PALETTE.ground);
}

/** Bands along one side, skipping the zero-area hinges between differently shaded points. */
const SIDE_BANDS: readonly [number, number][] = [
  [0, 1],
  [2, 3],
  [4, 5],
  [5, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [12, 13],
  [14, 15],
  [15, 16],
];

function layRoad(soup: Soup, line: StreetLine): void {
  const columns = ROAD_COLUMNS.length;
  const row = (): Point[] => Array.from({ length: columns }, point);
  let current = row();
  let upcoming = row();
  const fill = (index: number, points: Point[]): void => {
    const left = line.left[index]!;
    const right = line.right[index]!;
    const width = right - left;
    const presence = line.presence[index]!;
    const s = line.s[index]!;
    const trust = line.trust[index]!;
    const soften = 0.25 + 0.55 * (1 - trust);
    for (let column = 0; column < columns; column++) {
      const along = ROAD_COLUMNS[column]!;
      const target = points[column]!;
      const d = left + width * along;
      target.d = d;
      target.h = 0;
      target.out = 0;
      target.up = 1;
      roadTone(s, d, target.color);
      const fromLeft = d - left;
      const fromRight = right - d;
      const curbLeft = line.curbLeft[index]!;
      const curbRight = line.curbRight[index]!;
      // Water and grit gather along a curb. Where there's no curb the edge dissolves instead.
      const gutter =
        curbLeft * (1 - smoothstep(0, 0.55, fromLeft)) +
        curbRight * (1 - smoothstep(0, 0.55, fromRight));
      target.color.lerp(PALETTE.gutter, clamp(gutter, 0, 1) * 0.75);
      const raggedLeft = (1 - curbLeft) * (1 - smoothstep(0, 0.7, fromLeft));
      const raggedRight = (1 - curbRight) * (1 - smoothstep(0, 0.7, fromRight));
      target.color.lerp(PALETTE.shoulder, clamp(raggedLeft + raggedRight, 0, 1) * soften);
      target.color.lerp(PALETTE.shoulder, (1 - trust) * 0.18);
      target.color.lerp(PALETTE.ground, 1 - presence);
    }
  };
  fill(0, current);
  for (let index = 0; index + 1 < line.count; index++) {
    fill(index + 1, upcoming);
    soup.at(line.x[index]!, -line.y[index]!);
    for (let column = 0; column + 1 < columns; column++) {
      band(soup, line, index, 1, current[column]!, current[column + 1]!, upcoming[column]!, upcoming[column + 1]!);
    }
    [current, upcoming] = [upcoming, current];
  }
}

function layVerge(soup: Soup, line: StreetLine, side: Side): void {
  const row = (): Point[] => Array.from({ length: SECTION_POINTS }, point);
  let current = row();
  let upcoming = row();
  section(line, side, 0, current);
  for (let index = 0; index + 1 < line.count; index++) {
    section(line, side, index + 1, upcoming);
    soup.at(line.x[index]!, -line.y[index]!);
    for (const [a, b] of SIDE_BANDS) {
      band(soup, line, index, side.sign, current[a]!, current[b]!, upcoming[a]!, upcoming[b]!);
    }
    [current, upcoming] = [upcoming, current];
  }
}

/** Thin joints across the sidewalk, so the paving has a scale you can read from the car. */
function layJoints(soup: Soup, line: StreetLine, side: Side): void {
  let next = line.s[line.firstReal]!;
  const corner = [new Vector3(), new Vector3(), new Vector3(), new Vector3()] as const;
  for (let index = line.firstReal; index + 1 < line.lastReal; index++) {
    const s = line.s[index]!;
    if (s < next) continue;
    next = s + JOINT_SPACING;
    const walk = side.walk[index]!;
    if (walk < 0.8) continue;
    const { verge } = rises(side, index);
    const inner = side.edge[index]! + side.sign * Math.max(CURB_WIDTH * side.curb[index]!, 0.09);
    const outer = inner + side.sign * walk;
    const h = verge + JOINT_HEIGHT;
    const along = JOINT_WIDTH / 2;
    const place = (target: Vector3, d: number, forward: number): void => {
      target.set(
        line.x[index]! + line.rx[index]! * d + line.tx[index]! * forward,
        h,
        -(line.y[index]! + line.ry[index]! * d + line.ty[index]! * forward),
      );
    };
    const [low, high] = side.sign > 0 ? [inner, outer] : [outer, inner];
    place(corner[0], low, -along);
    place(corner[1], high, -along);
    place(corner[2], high, along);
    place(corner[3], low, along);
    soup.at(line.x[index]!, -line.y[index]!);
    soup.quad(corner[0], corner[1], corner[2], corner[3], PALETTE.sidewalkJoint);
  }
}

/** Lay paint flat on the road. Rings that won't triangulate are skipped, not drawn wrong. */
export function layPaint(soup: Soup, shapes: readonly PaintShape[]): number {
  let laid = 0;
  for (const shape of shapes) {
    const contour = shape.ring.map(([x, y]) => new Vector2(x, y));
    if (ShapeUtils.area(contour) > 0) contour.reverse();
    let faces: number[][];
    try {
      faces = ShapeUtils.triangulateShape(contour, []);
    } catch {
      continue;
    }
    if (!faces.length) continue;
    laid++;
    const first = contour[0]!;
    soup.at(first.x, -first.y);
    for (const [a, b, c] of faces) {
      const pa = contour[a!]!;
      const pb = contour[b!]!;
      const pc = contour[c!]!;
      soup.triangle(pa.x, PAINT_HEIGHT, -pa.y, pb.x, PAINT_HEIGHT, -pb.y, pc.x, PAINT_HEIGHT, -pc.y, PALETTE.paint);
    }
  }
  return laid;
}

/**
 * Uprights are placed from the video, so one standing inside the road's edges means the edge there
 * was a guess, not that a sign grew out of a lane. Real ones stand on a raised island, so give it one.
 */
export function layIslands(soup: Soup, objects: readonly StreetObject[], index: StreetIndex): number {
  const { line } = index;
  let laid = 0;
  const matrix = new Matrix4();
  const rotation = new Quaternion();
  const position = new Vector3();
  const scale = new Vector3();
  const axis = new Vector3(0, 1, 0);
  for (const object of objects) {
    const place = index.place(object.x, object.y);
    const station = place.station;
    const inside =
      place.d > line.left[station]! + ISLAND_MARGIN && place.d < line.right[station]! - ISLAND_MARGIN;
    if (!inside) continue;
    laid++;
    const plant = object.kind === "tree" || object.kind === "bush";
    const radius = plant ? ISLAND_RADIUS * 1.5 : ISLAND_RADIUS;
    const yaw = Math.atan2(line.tx[station]!, line.ty[station]!);
    rotation.setFromAxisAngle(axis, yaw);
    position.set(object.x, 0, -object.y);
    scale.set(1, 1, ISLAND_LENGTH);
    matrix.compose(position, rotation, scale);
    soup.at(object.x, -object.y);
    soup.add(islandCurb(radius), PALETTE.curb, matrix);
    soup.add(islandTop(radius), plant ? PALETTE.grass : PALETTE.sidewalk, matrix);
  }
  return laid;
}

function islandCurb(radius: number): BufferGeometry {
  const geometry = new CylinderGeometry(radius - 0.05, radius, ISLAND_RISE, 28, 1, true);
  geometry.translate(0, ISLAND_RISE / 2, 0);
  return geometry;
}

function islandTop(radius: number): BufferGeometry {
  const profile = [
    new Vector2(radius - 0.05, ISLAND_RISE),
    new Vector2(radius - 0.2, ISLAND_RISE + 0.012),
    new Vector2(0.001, ISLAND_RISE + 0.012),
  ];
  return new LatheGeometry(profile, 28);
}

export function laySurface(soup: Soup, line: StreetLine): void {
  layRoad(soup, line);
  for (const sign of [-1, 1] as const) {
    const side = sideOf(line, sign);
    layVerge(soup, line, side);
    layJoints(soup, line, side);
  }
}
