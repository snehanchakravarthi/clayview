import {
  BufferGeometry,
  CatmullRomCurve3,
  Color,
  CylinderGeometry,
  Matrix4,
  Quaternion,
  SphereGeometry,
  TubeGeometry,
  Vector2,
  Vector3,
} from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { cleanRun, type CleanRun, type RunOptions } from "../data/runs";
import type { BarrierRun, StreetObject, WorldPoint } from "../data/scene";
import { clamp, seeded, type StreetIndex } from "../data/street";
import { PALETTE } from "./palette";
import type { Soup } from "./soup";

const UP = new Vector3(0, 1, 0);
const MATRIX = new Matrix4();
const LOCAL = new Matrix4();
const QUATERNION = new Quaternion();
const SCALE_ONE = new Vector3(1, 1, 1);
const POSITION = new Vector3();

const POST_MAX_HEIGHT = 1.6;
const SIGN_MIN_HEIGHT = 2.5;
const SIGN_MAX_HEIGHT = 3.3;
const TALL_SIGN = 5.5;
const LIGHT_MIN_HEIGHT = 7.5;
const LIGHT_MAX_HEIGHT = 11;
const SIGNAL_MIN_HEIGHT = 5.2;
const SIGNAL_MAX_HEIGHT = 7;
const SIGNAL_MAX_REACH = 7.5;

const RUN_STEP = 3;

/** Where a part sits on its owner, with the owner facing `yaw` about the vertical. */
function placed(x: number, y: number, z: number, yaw = 0): Matrix4 {
  QUATERNION.setFromAxisAngle(UP, yaw);
  return LOCAL.compose(POSITION.set(x, y, z), QUATERNION, SCALE_ONE);
}

function column(top: number, bottom: number, height: number, sides = 10): BufferGeometry {
  const geometry = new CylinderGeometry(top, bottom, height, sides, 1);
  geometry.translate(0, height / 2, 0);
  geometry.deleteAttribute("uv");
  return geometry;
}

function box(width: number, height: number, depth: number, radius = 0.03): BufferGeometry {
  const geometry = new RoundedBoxGeometry(width, height, depth, 2, Math.min(radius, width / 2.5, height / 2.5, depth / 2.5));
  geometry.deleteAttribute("uv");
  return geometry;
}

function disc(radius: number, thickness: number, sides: number): BufferGeometry {
  const geometry = new CylinderGeometry(radius, radius, thickness, sides);
  geometry.rotateX(Math.PI / 2);
  geometry.deleteAttribute("uv");
  return geometry;
}

/** A vertical pole foot and shaft. */
function addMast(soup: Soup, base: Matrix4, height: number, radius: number, tone: Color): void {
  soup.add(column(radius * 0.55, radius, height, 12), tone, base);
  const collar = new Matrix4().multiplyMatrices(base, new Matrix4().makeTranslation(0, 0, 0));
  soup.add(column(radius * 1.25, radius * 1.45, 0.42, 12), PALETTE.steelDark, collar);
}

function addPost(soup: Soup, base: Matrix4, seed: number): void {
  const height = 1 + seeded(seed, 5) * 0.12;
  soup.add(column(0.06, 0.07, height, 8), PALETTE.panel, base);
  const band = new Matrix4().multiplyMatrices(base, placed(0, height - 0.3, 0));
  soup.add(column(0.066, 0.068, 0.15, 8), PALETTE.steelDark, band);
  const cap = new Matrix4().multiplyMatrices(base, placed(0, height, 0));
  soup.add(new SphereGeometry(0.06, 8, 5), PALETTE.panel, cap);
}

function addPole(soup: Soup, object: StreetObject, base: Matrix4, seed: number): void {
  if (object.height < POST_MAX_HEIGHT) {
    addPost(soup, base, seed);
    return;
  }
  addMast(soup, base, clamp(object.height, 3.5, 12), 0.1, PALETTE.steel);
}

/** A curved arm from a mast top out over the road, ending level. */
function arm(height: number, reach: number): BufferGeometry {
  const curve = new CatmullRomCurve3([
    new Vector3(0, height - 0.5, 0),
    new Vector3(0, height, -reach * 0.1),
    new Vector3(0, height + 0.32, -reach * 0.5),
    new Vector3(0, height + 0.42, -reach),
  ]);
  const geometry = new TubeGeometry(curve, 16, 0.042, 7, false);
  geometry.deleteAttribute("uv");
  return geometry;
}

function addStreetLight(soup: Soup, object: StreetObject, base: Matrix4, seed: number): void {
  const height = clamp(object.height, LIGHT_MIN_HEIGHT, LIGHT_MAX_HEIGHT);
  const reach = 1.5 + seeded(seed, 6) * 0.9;
  addMast(soup, base, height, 0.1, PALETTE.steel);
  soup.add(arm(height, reach), PALETTE.steel, base);
  const head = new Matrix4().multiplyMatrices(base, placed(0, height + 0.4, -reach - 0.22));
  soup.add(box(0.26, 0.09, 0.66, 0.04), PALETTE.steelDark, head);
  const lamp = new Matrix4().multiplyMatrices(base, placed(0, height + 0.35, -reach - 0.22));
  soup.add(box(0.18, 0.02, 0.42, 0.008), PALETTE.panel, lamp);
}

function addHousing(soup: Soup, at: Matrix4): void {
  soup.add(box(0.46, 1.16, 0.05, 0.02), PALETTE.steel, at);
  const housing = new Matrix4().multiplyMatrices(at, placed(0, 0, -0.1));
  soup.add(box(0.3, 0.96, 0.2, 0.05), PALETTE.steelDark, housing);
  for (const y of [-0.3, 0, 0.3]) {
    const lens = new Matrix4().multiplyMatrices(at, placed(0, y, -0.2));
    soup.add(disc(0.095, 0.03, 16), PALETTE.gutter, lens);
    const hood = new Matrix4().multiplyMatrices(at, placed(0, y + 0.11, -0.25));
    soup.add(box(0.22, 0.015, 0.14, 0.005), PALETTE.steelDark, hood);
  }
}

function addSignal(soup: Soup, object: StreetObject, base: Matrix4, reach: number): void {
  const height = clamp(object.height, SIGNAL_MIN_HEIGHT, SIGNAL_MAX_HEIGHT);
  addMast(soup, base, height + 0.5, 0.11, PALETTE.steel);
  // The mast arm reaches over the road. The housings face traffic coming along the road.
  const curve = new CatmullRomCurve3([
    new Vector3(0, height - 0.1, 0),
    new Vector3(0, height + 0.08, -0.4),
    new Vector3(0, height + 0.13, -reach * 0.5),
    new Vector3(0, height + 0.13, -reach),
  ]);
  const armGeometry = new TubeGeometry(curve, 14, 0.055, 7, false);
  armGeometry.deleteAttribute("uv");
  soup.add(armGeometry, PALETTE.steel, base);
  // The housing is built facing -z in its own frame; on the arm that's sideways, so turn it.
  const hung = new Matrix4().multiplyMatrices(base, placed(0, height - 0.48, -reach + 0.35, Math.PI / 2));
  addHousing(soup, hung);
  const low = new Matrix4().multiplyMatrices(base, placed(0.17, 2.9, 0, Math.PI / 2));
  addHousing(soup, low);
}

function addSign(soup: Soup, object: StreetObject, base: Matrix4, seed: number): void {
  const tall = object.height >= TALL_SIGN;
  const height = tall ? clamp(object.height, 5, 7.5) : clamp(object.height, SIGN_MIN_HEIGHT, SIGN_MAX_HEIGHT);
  addMast(soup, base, height, tall ? 0.09 : 0.045, PALETTE.steel);
  const front = new Matrix4().multiplyMatrices(base, placed(0, height - (tall ? 0.8 : 0.42), -0.05));
  const face = new Matrix4().multiplyMatrices(front, placed(0, 0, -0.016));
  if (tall) {
    soup.add(box(2.5, 1.5, 0.05, 0.03), PALETTE.steel, front);
    soup.add(box(2.36, 1.36, 0.02, 0.01), PALETTE.panel, face);
    return;
  }
  const form = Math.floor(seeded(seed, 7) * 4);
  if (form === 0) {
    soup.add(disc(0.43, 0.03, 32), PALETTE.steelDark, front);
    soup.add(disc(0.385, 0.02, 32), PALETTE.panel, face);
  } else if (form === 1) {
    soup.add(box(0.74, 0.92, 0.03, 0.03), PALETTE.steelDark, front);
    soup.add(box(0.66, 0.84, 0.02, 0.02), PALETTE.panel, face);
  } else if (form === 2) {
    const turned = new Matrix4().makeRotationZ(Math.PI / 4);
    soup.add(box(0.66, 0.66, 0.03, 0.03), PALETTE.steelDark, new Matrix4().multiplyMatrices(front, turned));
    soup.add(box(0.58, 0.58, 0.02, 0.02), PALETTE.panel, new Matrix4().multiplyMatrices(face, turned));
  } else {
    const point = new Matrix4().makeRotationZ(Math.PI);
    soup.add(disc(0.52, 0.03, 3), PALETTE.steelDark, new Matrix4().multiplyMatrices(front, point));
    soup.add(disc(0.45, 0.02, 3), PALETTE.panel, new Matrix4().multiplyMatrices(face, point));
  }
}

/** Build every pole, light, signal and sign of a shot. Returns how many were placed. */
export function layFurniture(soup: Soup, objects: readonly StreetObject[], index: StreetIndex): number {
  const { line } = index;
  let count = 0;
  for (const object of objects) {
    if (object.kind === "tree" || object.kind === "bush") continue;
    count++;
    const seed = object.x * 2.3 + object.y * 4.1;
    const place = index.place(object.x, object.y);
    const station = place.station;
    const rx = line.rx[station]!;
    const ry = line.ry[station]!;
    const tx = line.tx[station]!;
    const ty = line.ty[station]!;
    const sign = place.d >= 0 ? 1 : -1;
    // Towards the road, for arms. In three.js axes a world (x, y) direction is (x, 0, -y).
    const inward = Math.atan2(sign * rx, -sign * -ry);
    // Facing back along the road, for anything meant to be read by traffic.
    const facing = Math.atan2(tx, -ty) + Math.PI;
    const yaw = object.kind === "sign" ? facing + Math.PI : inward;
    const base = new Matrix4().compose(
      POSITION.set(object.x, 0, -object.y),
      QUATERNION.setFromAxisAngle(UP, yaw),
      SCALE_ONE,
    );
    soup.at(object.x, -object.y);
    if (object.kind === "pole") addPole(soup, object, base, seed);
    else if (object.kind === "streetLight") addStreetLight(soup, object, base, seed);
    else if (object.kind === "sign") addSign(soup, object, base, seed);
    else {
      const reach = clamp(Math.abs(place.d) * 0.72, 3, SIGNAL_MAX_REACH);
      addSignal(soup, object, base, reach);
    }
  }
  MATRIX.identity();
  return count;
}

type Profile = readonly (readonly [out: number, height: number])[];

interface Frame {
  x: number;
  z: number;
  /** Mitred outward vector at this point. */
  nx: number;
  nz: number;
}

/** Offset frames along a path: `n` points to the right of travel. */
function frames(points: readonly WorldPoint[]): Frame[] {
  const result: Frame[] = [];
  const count = points.length;
  const axis = new Vector2();
  const before = new Vector2();
  const after = new Vector2();
  for (let index = 0; index < count; index++) {
    const [x, y] = points[index]!;
    const [px, py] = points[Math.max(index - 1, 0)]!;
    const [nx, ny] = points[Math.min(index + 1, count - 1)]!;
    before.set(x - px, -(y - py)).normalize();
    after.set(nx - x, -(ny - y)).normalize();
    if (index === 0) before.copy(after);
    if (index === count - 1) after.copy(before);
    axis.addVectors(before, after).normalize();
    const miter = 1 / Math.max(axis.dot(after), 0.6);
    result.push({ x, z: -y, nx: -axis.y * miter, nz: axis.x * miter });
  }
  return result;
}

const CORNER_A = new Vector3();
const CORNER_B = new Vector3();
const CORNER_C = new Vector3();
const CORNER_D = new Vector3();

/** Sweep a counter-clockwise profile (facing along the path) down a path. */
function sweep(soup: Soup, points: readonly WorldPoint[], profile: Profile, color: Color, shift = 0, heightAt?: (index: number) => number): void {
  const path = frames(points);
  const corners = profile.length;
  const lift = heightAt ?? ((): number => 1);
  const set = (target: Vector3, frame: Frame, corner: number, rise: number): void => {
    const [out, height] = profile[corner]!;
    target.set(frame.x + frame.nx * (out + shift), height * rise, frame.z + frame.nz * (out + shift));
  };
  for (let index = 0; index + 1 < path.length; index++) {
    const from = path[index]!;
    const to = path[index + 1]!;
    soup.at(from.x, from.z);
    for (let corner = 0; corner < corners; corner++) {
      const next = (corner + 1) % corners;
      set(CORNER_A, from, corner, lift(index));
      set(CORNER_B, from, next, lift(index));
      set(CORNER_C, to, next, lift(index + 1));
      set(CORNER_D, to, corner, lift(index + 1));
      soup.quad(CORNER_A, CORNER_D, CORNER_C, CORNER_B, color);
    }
  }
}

/** Close an end of a swept profile with a fan. */
function capEnds(soup: Soup, points: readonly WorldPoint[], profile: Profile, color: Color): void {
  const path = frames(points);
  const ends: [Frame, boolean][] = [
    [path[0]!, true],
    [path[path.length - 1]!, false],
  ];
  for (const [frame, first] of ends) {
    soup.at(frame.x, frame.z);
    const count = profile.length;
    for (let corner = 1; corner + 1 < count; corner++) {
      const [a, b, c] = first
        ? [profile[0]!, profile[corner]!, profile[corner + 1]!]
        : [profile[0]!, profile[corner + 1]!, profile[corner]!];
      soup.triangle(
        frame.x + frame.nx * a[0], a[1], frame.z + frame.nz * a[0],
        frame.x + frame.nx * b[0], b[1], frame.z + frame.nz * b[0],
        frame.x + frame.nx * c[0], c[1], frame.z + frame.nz * c[0],
        color,
      );
    }
  }
}

/** Evenly spaced places along a path, with the direction of travel there. */
function stations(points: readonly WorldPoint[], spacing: number, visit: (x: number, z: number, yaw: number) => void): void {
  let run = spacing / 2;
  for (let index = 0; index + 1 < points.length; index++) {
    const [ax, ay] = points[index]!;
    const [bx, by] = points[index + 1]!;
    const length = Math.hypot(bx - ax, by - ay);
    while (run <= length) {
      const mix = run / length;
      visit(ax + (bx - ax) * mix, -(ay + (by - ay) * mix), Math.atan2(bx - ax, by - ay));
      run += spacing;
    }
    run -= length;
  }
}

function layWall(soup: Soup, run: CleanRun, height: number, seed: number): void {
  const half = 0.13;
  const cap = 0.19;
  const body: Profile = [
    [-half, 0],
    [half, 0],
    [half, height - 0.07],
    [-half, height - 0.07],
  ];
  sweep(soup, run.points, body, PALETTE.wall);
  capEnds(soup, run.points, body, PALETTE.wall);
  const coping: Profile = [
    [-cap, height - 0.09],
    [cap, height - 0.09],
    [cap, height - 0.03],
    [cap - 0.05, height],
    [-cap + 0.05, height],
    [-cap, height - 0.03],
  ];
  sweep(soup, run.points, coping, PALETTE.concrete);
  capEnds(soup, run.points, coping, PALETTE.concrete);
  const pier = box(0.44, height + 0.12, 0.44, 0.03);
  pier.translate(0, (height + 0.12) / 2, 0);
  const spacing = 6 + seeded(seed, 8) * 3;
  stations(run.points, spacing, (x, z, yaw) => {
    soup.at(x, z).add(pier, PALETTE.concrete, placed(x, 0, z, yaw));
  });
}

function layFence(soup: Soup, run: CleanRun, height: number): void {
  const rail = (y: number): Profile => [
    [-0.022, y - 0.024],
    [0.022, y - 0.024],
    [0.022, y + 0.024],
    [-0.022, y + 0.024],
  ];
  sweep(soup, run.points, rail(height - 0.14), PALETTE.steelDark);
  sweep(soup, run.points, rail(0.16), PALETTE.steelDark);
  const post = column(0.032, 0.036, height, 6);
  stations(run.points, 2.5, (x, z, yaw) => {
    soup.at(x, z).add(post, PALETTE.steelDark, placed(x, 0, z, yaw));
  });
  const picket = box(0.018, height - 0.16, 0.05, 0.006);
  picket.translate(0, 0.08 + (height - 0.16) / 2, 0);
  stations(run.points, 0.2, (x, z, yaw) => {
    soup.at(x, z).add(picket, PALETTE.steel, placed(x, 0, z, yaw));
  });
}

function layGuardRail(soup: Soup, run: CleanRun, side: number): void {
  const beam = 0.64;
  // The W of a real rail: two ridges facing the road.
  const facing = -side;
  const section = (out: number, y: number): readonly [number, number] => [out * facing, beam + y];
  const ribbonOutline: Profile = (
    [
      section(0.06, -0.16),
      section(0.11, -0.1),
      section(0.05, -0.02),
      section(0.05, 0.02),
      section(0.11, 0.1),
      section(0.06, 0.16),
      section(0.02, 0.16),
      section(0.08, 0.1),
      section(0.015, 0.02),
      section(0.015, -0.02),
      section(0.08, -0.1),
      section(0.02, -0.16),
    ] as const
  );
  const ordered = facing > 0 ? [...ribbonOutline].reverse() : ribbonOutline;
  sweep(soup, run.points, ordered, PALETTE.steel);
  const post = box(0.13, beam + 0.08, 0.06, 0.012);
  post.translate(0, (beam + 0.08) / 2, 0);
  const block = box(0.1, 0.16, 0.08, 0.01);
  stations(run.points, 2.25, (x, z, yaw) => {
    soup.at(x, z).add(post, PALETTE.steelDark, placed(x, 0, z, yaw));
  });
  void block;
}

function layJersey(soup: Soup, run: CleanRun, height: number): void {
  const h = height;
  const profile: Profile = [
    [-0.32, 0],
    [0.32, 0],
    [0.32, h * 0.1],
    [0.16, h * 0.3],
    [0.09, h * 0.92],
    [0.05, h],
    [-0.05, h],
    [-0.09, h * 0.92],
    [-0.16, h * 0.3],
    [-0.32, h * 0.1],
  ];
  sweep(soup, run.points, profile, PALETTE.concrete);
  capEnds(soup, run.points, profile, PALETTE.concrete);
}

const BARRIER_RULES: Record<BarrierRun["kind"], RunOptions> = {
  wall: { splitJump: 3.5, splitGap: 14, minLength: 4, clearance: 0.25, behindVerge: true, step: RUN_STEP },
  fence: { splitJump: 3, splitGap: 14, minLength: 3, clearance: 0.25, behindVerge: true, step: RUN_STEP },
  guardRail: { splitJump: 2.5, splitGap: 14, minLength: 6, clearance: 0.5, behindVerge: false, step: RUN_STEP },
  barrier: { splitJump: 2.5, splitGap: 14, minLength: 3, clearance: 0.35, behindVerge: false, step: RUN_STEP },
};

/** Build walls, fences, guard rails and barriers. Returns how many runs survived cleaning. */
export function layBarriers(soup: Soup, barriers: readonly BarrierRun[], index: StreetIndex): number {
  let laid = 0;
  for (const barrier of barriers) {
    const side = barrier.side === "right" ? 1 : -1;
    const seed = barrier.points[0]![0] + barrier.points[0]![1];
    for (const run of cleanRun(barrier.points, barrier.side, index, BARRIER_RULES[barrier.kind])) {
      laid++;
      switch (barrier.kind) {
        case "wall":
          layWall(soup, run, clamp(barrier.height, 1.1, 2.6), seed);
          break;
        case "fence":
          layFence(soup, run, clamp(barrier.height, 1.2, 2.1));
          break;
        case "guardRail":
          layGuardRail(soup, run, side);
          break;
        case "barrier":
          layJersey(soup, run, clamp(barrier.height, 0.8, 1.1));
          break;
      }
    }
  }
  return laid;
}
