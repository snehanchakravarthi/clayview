import {
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DynamicDrawUsage,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  Matrix4,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from "three";
import { mergeGeometries, mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";
import type { StreetObject } from "../data/scene";
import { clamp, seeded, smoothstep } from "../data/street";
import { withCutaway } from "./cutaway";
import { PALETTE } from "./palette";

const NEAR_RANGE = 74;
const FAR_RANGE = 240;
const TREE_VARIANTS = 6;
const BUSH_VARIANTS = 3;
const NEAR_DETAIL = 13;
const FAR_DETAIL = 3;

const UP = new Vector3(0, 1, 0);
const MATRIX = new Matrix4();
const POSITION = new Vector3();
const ROTATION = new Quaternion();
const SCALE = new Vector3();
const TINT = new Color();

/** Instances are tinted for their leaves, so bark is drawn brighter to come out as bark. */
const BARK = new Color().setRGB(
  PALETTE.bark.r / PALETTE.leaf.r,
  PALETTE.bark.g / PALETTE.leaf.g,
  PALETTE.bark.b / PALETTE.leaf.b,
);

interface Lobe {
  x: number;
  y: number;
  z: number;
  radius: number;
}

/** Soft-maximum of the distance out to each lobe, so lobes melt together with a crease. */
function lobeReach(lobes: readonly Lobe[], dx: number, dy: number, dz: number): number {
  let sum = 0;
  const sharpness = 24;
  for (const lobe of lobes) {
    const along = dx * lobe.x + dy * lobe.y + dz * lobe.z;
    const square = lobe.x * lobe.x + lobe.y * lobe.y + lobe.z * lobe.z;
    const inside = lobe.radius * lobe.radius - square + along * along;
    if (inside <= 0) continue;
    const reach = along + Math.sqrt(inside);
    if (reach > 0) sum += Math.exp(sharpness * reach);
  }
  return sum > 0 ? Math.log(sum) / sharpness : 0.2;
}

/** Value noise, for the small bumps a thumb leaves in clay. */
function bumps(x: number, y: number, z: number, seed: number): number {
  const fx = Math.floor(x);
  const fy = Math.floor(y);
  const fz = Math.floor(z);
  const tx = x - fx;
  const ty = y - fy;
  const tz = z - fz;
  const sx = tx * tx * (3 - 2 * tx);
  const sy = ty * ty * (3 - 2 * ty);
  const sz = tz * tz * (3 - 2 * tz);
  const at = (a: number, b: number, c: number): number =>
    seeded(fx + a + fz * 57 + c * 57, fy + b + seed, seed);
  const row = (b: number, c: number): number => at(0, b, c) + (at(1, b, c) - at(0, b, c)) * sx;
  const sheet = (c: number): number => row(0, c) + (row(1, c) - row(0, c)) * sy;
  return sheet(0) + (sheet(1) - sheet(0)) * sz;
}

interface Crown {
  lobes: number;
  spread: number;
  squash: number;
  bump: number;
  lobeSize: number;
}

/**
 * A crown of foliage: one continuous star-shaped skin, so there's no hidden geometry inside it.
 * Creases between lobes are darkened in the vertex colours, because the crown's own shadowing is
 * far bigger than the AO radius. Unit crown: about one radius wide, sitting around the origin.
 */
function crown(seed: number, detail: number, shape: Crown): BufferGeometry {
  const lobes: Lobe[] = [];
  for (let index = 0; index < shape.lobes; index++) {
    const angle = index * 2.39996 + seeded(seed, index, 1) * 1.1;
    // Golden-angle spiral with a little scatter, so lobes cover the crown instead of clumping.
    const rise = 1 - ((index + 0.5) / shape.lobes) * 1.75 + (seeded(seed, index, 2) - 0.5) * 0.18;
    const flat = Math.sqrt(Math.max(1 - rise * rise, 0.02));
    const out = shape.spread * (0.82 + seeded(seed, index, 3) * 0.3);
    lobes.push({
      x: Math.cos(angle) * flat * out,
      y: rise * out * 0.9,
      z: Math.sin(angle) * flat * out,
      radius: shape.lobeSize * (0.68 + seeded(seed, index, 4) * 0.62),
    });
  }
  lobes.push({ x: 0, y: 0.05, z: 0, radius: 0.62 });

  const base = new IcosahedronGeometry(1, detail);
  base.deleteAttribute("uv");
  base.deleteAttribute("normal");
  const geometry = mergeVertices(base, 1e-4);
  const position = geometry.getAttribute("position") as BufferAttribute;
  const shades = new Float32Array(position.count * 3);
  let largest = 0;
  const reaches = new Float32Array(position.count);
  for (let index = 0; index < position.count; index++) {
    const reach = lobeReach(lobes, position.getX(index), position.getY(index), position.getZ(index));
    reaches[index] = reach;
    largest = Math.max(largest, reach);
  }
  for (let index = 0; index < position.count; index++) {
    const dx = position.getX(index);
    const dy = position.getY(index);
    const dz = position.getZ(index);
    const noise =
      bumps(dx * 5 + seed, dy * 5, dz * 5, seed) * 0.6 +
      bumps(dx * 11 + seed, dy * 11, dz * 11, seed + 1) * 0.4;
    const reach = reaches[index]! / largest;
    const radius = reach * (1 + (noise - 0.5) * shape.bump * (detail > 6 ? 1 : 0.55));
    // Crowns hang: flatter underneath than on top.
    const under = dy < 0 ? 1 - 0.32 * smoothstep(0, 1, -dy) : 1;
    position.setXYZ(index, dx * radius, dy * radius * under * shape.squash, dz * radius);
    const crease = smoothstep(0.66, 0.97, reach);
    const light = smoothstep(-0.9, 0.55, dy);
    const shade = (0.66 + 0.42 * light) * (0.68 + 0.36 * crease) * (0.9 + 0.2 * noise);
    shades[index * 3] = shade;
    shades[index * 3 + 1] = shade;
    shades[index * 3 + 2] = shade;
  }
  geometry.setAttribute("color", new BufferAttribute(shades, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function painted(geometry: BufferGeometry, color: Color): BufferGeometry {
  const count = geometry.getAttribute("position").count;
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index++) {
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new BufferAttribute(colors, 3));
  return geometry;
}

/** A tapered limb between two points, with its colour painted on. */
function limb(from: Vector3, to: Vector3, bottom: number, top: number, sides: number): BufferGeometry {
  const direction = to.clone().sub(from);
  const length = direction.length();
  const geometry = new CylinderGeometry(top, bottom, length, sides, 1, true);
  geometry.deleteAttribute("uv");
  geometry.applyQuaternion(new Quaternion().setFromUnitVectors(UP, direction.normalize()));
  const middle = from.clone().add(to).multiplyScalar(0.5);
  geometry.translate(middle.x, middle.y, middle.z);
  return painted(geometry, BARK);
}

/**
 * A tree one unit tall and one unit in crown radius, stretched per instance. Height and width come
 * apart at that point, so the trunk is drawn thin and thick enough for the usual proportions.
 */
function treeGeometry(seed: number, detail: number): BufferGeometry {
  const near = detail > 6;
  const sides = near ? 9 : 5;
  const crownBase = 0.2 + seeded(seed, 11) * 0.12;
  const lean = (seeded(seed, 12) - 0.5) * 0.16;
  const leanZ = (seeded(seed, 13) - 0.5) * 0.16;
  const fork = new Vector3(lean, crownBase + 0.12, leanZ);
  const foot = new Vector3(lean * 0.12, 0.022, leanZ * 0.12);
  const parts: BufferGeometry[] = [
    limb(new Vector3(0, -0.006, 0), foot, 0.105, 0.072, sides),
    limb(foot, fork, 0.072, 0.048, sides),
  ];
  const limbs = near ? 3 : 2;
  for (let index = 0; index < limbs; index++) {
    const angle = seeded(seed, index, 14) * Math.PI * 2 + (index * Math.PI * 2) / limbs;
    const tip = new Vector3(
      fork.x + Math.cos(angle) * 0.42,
      fork.y + 0.18 + seeded(seed, index, 15) * 0.12,
      fork.z + Math.sin(angle) * 0.42,
    );
    parts.push(limb(fork, tip, 0.05, 0.014, near ? 6 : 4));
  }
  const canopy = crown(seed, detail, {
    lobes: 15 + Math.floor(seeded(seed, 17) * 9),
    spread: 0.68,
    squash: 1,
    bump: 0.13,
    lobeSize: 0.33 + seeded(seed, 16) * 0.08,
  });
  // Unit width, but only the top of the unit height: the crown is squeezed to fit.
  const middle = crownBase + (1 - crownBase) * 0.5;
  const halfHeight = (1 - crownBase) * 0.5;
  canopy.scale(1, halfHeight, 1);
  canopy.translate(0, middle, 0);
  canopy.computeVertexNormals();
  const merged = mergeGeometries(
    [...parts.map((part) => (part.index ? part.toNonIndexed() : part)), canopy.toNonIndexed()],
    false,
  );
  if (!merged) throw new Error("could not merge a tree");
  return merged;
}

function bushGeometry(seed: number, detail: number): BufferGeometry {
  const geometry = crown(seed + 50, detail, {
    lobes: 14,
    spread: 0.66,
    squash: 0.82,
    bump: 0.2,
    lobeSize: 0.36,
  });
  geometry.scale(1, 0.5, 1);
  geometry.translate(0, 0.42, 0);
  geometry.computeVertexNormals();
  return geometry;
}

interface Planted {
  x: number;
  z: number;
  rotation: number;
  width: number;
  height: number;
  variant: number;
  tint: Color;
}

function tintFor(seed: number, rich: boolean): Color {
  const color = new Color().copy(PALETTE.leaf);
  color.lerp(PALETTE.leafDeep, seeded(seed, 21) * (rich ? 0.8 : 0.5));
  color.lerp(PALETTE.leafWarm, seeded(seed, 22) * 0.45);
  return color;
}

/** Tree heights run low when the crown left the frame, so trust the shape of a tree over them. */
export function treeSize(object: StreetObject): { height: number; width: number } {
  const seed = object.x * 3.1 + object.y * 1.7;
  const measured = object.radius ?? 2;
  const height = clamp(Math.max(object.height, 8.5 + seeded(seed, 1) * 5), 7, 21);
  const width = clamp(Math.max(measured * 1.1, height * (0.27 + seeded(seed, 2) * 0.1)), 2.4, 7.5);
  return { height, width };
}

function plant(objects: readonly StreetObject[]): { trees: Planted[]; bushes: Planted[] } {
  const trees: Planted[] = [];
  const bushes: Planted[] = [];
  for (const object of objects) {
    const seed = object.x * 3.1 + object.y * 1.7;
    if (object.kind === "tree") {
      const { height, width } = treeSize(object);
      trees.push({
        x: object.x,
        z: -object.y,
        rotation: seeded(seed, 3) * Math.PI * 2,
        width,
        height,
        variant: Math.floor(seeded(seed, 4) * TREE_VARIANTS),
        tint: tintFor(seed, true),
      });
    } else if (object.kind === "bush") {
      const width = clamp((object.radius ?? 1) * 1.15, 0.8, 2.4);
      bushes.push({
        x: object.x,
        z: -object.y,
        rotation: seeded(seed, 3) * Math.PI * 2,
        width,
        height: clamp(object.height * 0.7, 1, 2.6),
        variant: Math.floor(seeded(seed, 4) * BUSH_VARIANTS),
        tint: tintFor(seed, false),
      });
    }
  }
  return { trees, bushes };
}

/** Shapes shared by every world, built once. */
interface Library {
  trees: { near: BufferGeometry; far: BufferGeometry }[];
  bushes: { near: BufferGeometry; far: BufferGeometry }[];
  material: MeshStandardMaterial;
}

let library: Library | null = null;

function shapes(): Library {
  if (library) return library;
  const variants = (
    count: number,
    make: (seed: number, detail: number) => BufferGeometry,
  ): { near: BufferGeometry; far: BufferGeometry }[] =>
    Array.from({ length: count }, (_, index) => ({
      near: make(index + 1, NEAR_DETAIL),
      far: make(index + 1, FAR_DETAIL),
    }));
  library = {
    trees: variants(TREE_VARIANTS, treeGeometry),
    bushes: variants(BUSH_VARIANTS, bushGeometry),
    material: withCutaway(
      new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }),
      true,
    ),
  };
  return library;
}

function instanced(geometry: BufferGeometry, material: MeshStandardMaterial, capacity: number): InstancedMesh {
  const mesh = new InstancedMesh(geometry, material, Math.max(capacity, 1));
  mesh.instanceMatrix.setUsage(DynamicDrawUsage);
  mesh.setColorAt(0, TINT.set(1, 1, 1));
  mesh.instanceColor?.setUsage(DynamicDrawUsage);
  mesh.count = 0;
  // Instances are chosen by distance every frame, so a stale bound must not cull them.
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.matrixAutoUpdate = false;
  return mesh;
}

interface Rack {
  planted: Planted[];
  near: InstancedMesh[];
  far: InstancedMesh[];
}

/**
 * Trees and bushes of one shot. Each frame they're dealt into near and far instanced meshes by
 * distance from ego, so detail goes where the camera is and nothing past the fog is drawn at all.
 */
export class Flora {
  readonly group = new Group();
  readonly treeCount: number;
  readonly bushCount: number;
  private readonly racks: Rack[];
  private lastX = Infinity;
  private lastY = Infinity;

  constructor(objects: readonly StreetObject[]) {
    const { trees, bushes } = plant(objects);
    this.treeCount = trees.length;
    this.bushCount = bushes.length;
    const library = shapes();
    const rack = (
      planted: Planted[],
      variants: { near: BufferGeometry; far: BufferGeometry }[],
    ): Rack[] =>
      variants.map((variant, index) => {
        const members = planted.filter((item) => item.variant === index);
        const near = instanced(variant.near, library.material, members.length);
        const far = instanced(variant.far, library.material, members.length);
        this.group.add(near, far);
        return { planted: members, near: [near], far: [far] };
      });
    this.racks = [...rack(trees, library.trees), ...rack(bushes, library.bushes)];
  }

  /** Re-deal for ego's position in the shot's world. Skipped until it has moved a little. */
  update(x: number, y: number): void {
    if (Math.hypot(x - this.lastX, y - this.lastY) < 1.5) return;
    this.lastX = x;
    this.lastY = y;
    const z = -y;
    for (const rack of this.racks) {
      const near = rack.near[0]!;
      const far = rack.far[0]!;
      let nearCount = 0;
      let farCount = 0;
      for (const item of rack.planted) {
        const range = Math.hypot(item.x - x, item.z - z);
        if (range > FAR_RANGE) continue;
        POSITION.set(item.x, 0, item.z);
        ROTATION.setFromAxisAngle(UP, item.rotation);
        SCALE.set(item.width, item.height, item.width);
        MATRIX.compose(POSITION, ROTATION, SCALE);
        if (range < NEAR_RANGE) {
          near.setMatrixAt(nearCount, MATRIX);
          near.setColorAt(nearCount, item.tint);
          nearCount++;
        } else {
          far.setMatrixAt(farCount, MATRIX);
          far.setColorAt(farCount, item.tint);
          farCount++;
        }
      }
      near.count = nearCount;
      far.count = farCount;
      near.instanceMatrix.needsUpdate = true;
      far.instanceMatrix.needsUpdate = true;
      if (near.instanceColor) near.instanceColor.needsUpdate = true;
      if (far.instanceColor) far.instanceColor.needsUpdate = true;
    }
  }

  /** Triangles drawn right now. */
  triangles(): number {
    let total = 0;
    for (const rack of this.racks) {
      for (const mesh of [rack.near[0]!, rack.far[0]!]) {
        total += (mesh.geometry.getAttribute("position").count / 3) * mesh.count;
      }
    }
    return total;
  }
}
