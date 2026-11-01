/**
 * Procedural clay models. No assets: every form is lofted or turned in code.
 * Models sit on y = 0 with the nose towards -z, and are built at real scale in metres.
 */

import {
  BufferGeometry,
  CapsuleGeometry,
  CylinderGeometry,
  LatheGeometry,
  Quaternion,
  SphereGeometry,
  TorusGeometry,
  Vector2,
  Vector3,
} from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { ObjectClass } from "../data/scene";
import { loft, type Station } from "./loft";

/** Three tones are all the palette a clay model needs. */
export type Tone = "body" | "glass" | "tyre";

export type ModelParts = Partial<Record<Tone, BufferGeometry>>;

type PartList = Partial<Record<Tone, BufferGeometry[]>>;

function merged(parts: PartList): ModelParts {
  const result: ModelParts = {};
  for (const tone of ["body", "glass", "tyre"] as const) {
    const list = parts[tone];
    if (!list?.length) continue;
    // Lathes and capsules are indexed, rounded boxes aren't; merging needs them all the same.
    const prepared = list.map((geometry) => {
      const flat = geometry.index ? geometry.toNonIndexed() : geometry;
      flat.deleteAttribute("uv");
      return flat;
    });
    const geometry = mergeGeometries(prepared, false);
    if (!geometry) throw new Error(`could not merge ${tone} parts`);
    result[tone] = geometry;
  }
  return result;
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** A tyre with a soft shoulder and a dished hub, turned around the axle. */
function wheel(radius: number, width: number, x: number, y: number, z: number): BufferGeometry {
  const half = width / 2;
  const hub = radius * 0.62;
  const profile = [
    new Vector2(0.001, half * 0.45),
    new Vector2(hub * 0.5, half * 0.48),
    new Vector2(hub, half * 0.72),
    new Vector2(radius * 0.8, half * 0.98),
    new Vector2(radius * 0.94, half),
    new Vector2(radius, half * 0.72),
    new Vector2(radius, -half * 0.72),
    new Vector2(radius * 0.94, -half),
    new Vector2(radius * 0.8, -half * 0.98),
    new Vector2(hub, -half * 0.72),
    new Vector2(hub * 0.5, -half * 0.48),
    new Vector2(0.001, -half * 0.45),
  ];
  const geometry = new LatheGeometry(profile, 32);
  geometry.rotateZ(Math.PI / 2);
  geometry.translate(x, y, z);
  return geometry;
}

/** Bore a wheel well into the flank of a lofted body. */
function archDeform(
  arches: readonly { z: number; y: number; radius: number }[],
  depth: number,
): (vertex: Vector3) => void {
  return (vertex) => {
    for (const arch of arches) {
      const distance = Math.hypot(vertex.z - arch.z, vertex.y - arch.y);
      const inside = 1 - smoothstep(arch.radius - 0.02, arch.radius + 0.05, distance);
      if (inside > 0 && Math.abs(vertex.x) > 0.2) {
        vertex.x -= Math.sign(vertex.x) * depth * inside;
      }
    }
  };
}

function limb(from: Vector3, to: Vector3, radius: number): BufferGeometry {
  const direction = to.clone().sub(from);
  const length = direction.length();
  const geometry = new CapsuleGeometry(radius, length, 6, 14);
  const rotation = new Quaternion().setFromUnitVectors(
    new Vector3(0, 1, 0),
    direction.clone().normalize(),
  );
  geometry.applyQuaternion(rotation);
  const middle = from.clone().add(to).multiplyScalar(0.5);
  geometry.translate(middle.x, middle.y, middle.z);
  return geometry;
}

function ball(radius: number, x: number, y: number, z: number): BufferGeometry {
  const geometry = new SphereGeometry(radius, 24, 16);
  geometry.translate(x, y, z);
  return geometry;
}

function roundedBox(
  width: number,
  height: number,
  depth: number,
  x: number,
  y: number,
  z: number,
  radius = 0.08,
): BufferGeometry {
  const geometry = new RoundedBoxGeometry(width, height, depth, 4, radius);
  geometry.translate(x, y, z);
  return geometry;
}

interface CarShape {
  length: number;
  halfWidth: number;
  deck: number;
  roof: number;
  /** Where the roof falls to the tail, in metres behind the centre. */
  rearFall: number;
  /** Where the windshield meets the hood. */
  cowl: number;
  ride: number;
  wheelRadius: number;
}

export const SEDAN: CarShape = {
  length: 4.6,
  halfWidth: 0.91,
  deck: 0.95,
  roof: 1.43,
  rearFall: 1.75,
  cowl: -1.32,
  ride: 0.2,
  wheelRadius: 0.33,
};

const HATCH: CarShape = {
  length: 4.2,
  halfWidth: 0.89,
  deck: 0.98,
  roof: 1.47,
  rearFall: 1.98,
  cowl: -1.22,
  ride: 0.2,
  wheelRadius: 0.32,
};

const SUV: CarShape = {
  length: 4.7,
  halfWidth: 0.95,
  deck: 1.1,
  roof: 1.72,
  rearFall: 2.2,
  cowl: -1.38,
  ride: 0.3,
  wheelRadius: 0.38,
};

function carParts(shape: CarShape): ModelParts {
  const half = shape.length / 2;
  const { halfWidth: w, deck, roof, ride, cowl, rearFall, wheelRadius } = shape;
  const axleFront = -half * 0.64;
  const axleRear = half * 0.66;
  const arches = [axleFront, axleRear].map((z) => ({
    z,
    y: wheelRadius,
    radius: wheelRadius + 0.075,
  }));

  const body: Station[] = [
    { z: half + 0.02, halfWidth: w * 0.55, top: deck - 0.16, bottom: ride + 0.26, square: 3.2 },
    { z: half - 0.06, halfWidth: w * 0.84, top: deck - 0.05, bottom: ride + 0.1, square: 3.2 },
    { z: half - 0.32, halfWidth: w * 0.96, top: deck + 0.01, bottom: ride + 0.04, square: 3.4 },
    { z: half * 0.6, halfWidth: w, top: deck + 0.02, bottom: ride, square: 3.6 },
    { z: 0, halfWidth: w, top: deck, bottom: ride, square: 3.6 },
    { z: cowl * 0.55, halfWidth: w, top: deck - 0.02, bottom: ride, square: 3.6 },
    { z: cowl, halfWidth: w * 0.99, top: deck - 0.1, bottom: ride, square: 3.4 },
    { z: -half * 0.84, halfWidth: w * 0.94, top: deck - 0.21, bottom: ride + 0.03, square: 3.2 },
    { z: -half * 0.97, halfWidth: w * 0.8, top: deck - 0.3, bottom: ride + 0.08, square: 3 },
    { z: -half - 0.02, halfWidth: w * 0.52, top: deck - 0.38, bottom: ride + 0.16, square: 3 },
  ];

  const roofBack = rearFall * 0.42;
  const roofFront = cowl * 0.3;
  const glassBase = deck - 0.14;
  const glass: Station[] = [
    { z: rearFall + 0.1, halfWidth: w * 0.5, top: deck + 0.02, bottom: glassBase, square: 2.8 },
    {
      z: rearFall - 0.18,
      halfWidth: w * 0.72,
      top: deck + (roof - deck) * 0.42,
      bottom: glassBase,
      square: 2.8,
      lean: 0.12,
    },
    {
      z: roofBack + (rearFall - roofBack) * 0.38,
      halfWidth: w * 0.79,
      top: roof - 0.05,
      bottom: glassBase,
      square: 3,
      lean: 0.16,
    },
    { z: roofBack, halfWidth: w * 0.8, top: roof, bottom: glassBase, square: 3, lean: 0.17 },
    { z: roofFront, halfWidth: w * 0.8, top: roof - 0.01, bottom: glassBase, square: 3, lean: 0.17 },
    {
      z: roofFront + (cowl - roofFront) * 0.5,
      halfWidth: w * 0.78,
      top: deck + (roof - deck) * 0.6,
      bottom: glassBase,
      square: 2.9,
      lean: 0.14,
    },
    { z: cowl + 0.04, halfWidth: w * 0.7, top: deck + 0.05, bottom: glassBase, square: 2.8 },
    { z: cowl - 0.2, halfWidth: w * 0.45, top: deck - 0.09, bottom: glassBase, square: 2.8 },
  ];

  const skinBase = roof - 0.11;
  const skinTail = roofBack + (rearFall - roofBack) * 0.56;
  const skinNose = roofFront + (cowl - roofFront) * 0.26;
  const skin: Station[] = [
    { z: skinTail + 0.03, halfWidth: w * 0.42, top: skinBase + 0.015, bottom: skinBase, square: 3 },
    { z: skinTail - 0.1, halfWidth: w * 0.68, top: roof - 0.05, bottom: skinBase, square: 3 },
    { z: roofBack, halfWidth: w * 0.76, top: roof + 0.018, bottom: skinBase, square: 3.2, lean: 0.06 },
    {
      z: roofFront,
      halfWidth: w * 0.76,
      top: roof + 0.012,
      bottom: skinBase,
      square: 3.2,
      lean: 0.06,
    },
    { z: skinNose + 0.1, halfWidth: w * 0.68, top: roof - 0.08, bottom: skinBase, square: 3 },
    { z: skinNose, halfWidth: w * 0.42, top: skinBase + 0.015, bottom: skinBase, square: 3 },
  ];

  const mirrorZ = cowl + 0.32;
  const mirrorY = deck + 0.03;
  const mirrors = [-1, 1].map((side) => {
    const mirror = new SphereGeometry(0.075, 16, 12);
    mirror.scale(1.45, 0.82, 0.8);
    mirror.translate(side * (w + 0.06), mirrorY, mirrorZ);
    return mirror;
  });

  const wheels = [axleFront, axleRear].flatMap((z) =>
    [-1, 1].map((side) => wheel(wheelRadius, 0.25, side * (w - 0.13), wheelRadius, z)),
  );

  return merged({
    body: [
      loft(body, {
        rings: 80,
        segments: 60,
        deform: archDeform(arches, 0.3),
      }),
      loft(skin, { rings: 28, segments: 28 }),
      ...mirrors,
    ],
    glass: [loft(glass, { rings: 48, segments: 36 })],
    tyre: wheels,
  });
}

function truckParts(): ModelParts {
  const width = 2.46;
  const half = width / 2;
  const cabZ = -3.0;
  const cab: Station[] = [
    { z: cabZ + 1.05, halfWidth: half * 0.8, top: 2.7, bottom: 0.7, square: 5 },
    { z: cabZ + 0.95, halfWidth: half, top: 2.86, bottom: 0.62, square: 6 },
    { z: cabZ + 0.2, halfWidth: half, top: 2.9, bottom: 0.6, square: 6 },
    { z: cabZ - 0.45, halfWidth: half, top: 2.82, bottom: 0.6, square: 6 },
    { z: cabZ - 0.86, halfWidth: half * 0.98, top: 2.4, bottom: 0.62, square: 5 },
    { z: cabZ - 0.98, halfWidth: half * 0.9, top: 1.5, bottom: 0.7, square: 5 },
    { z: cabZ - 1.0, halfWidth: half * 0.6, top: 1.0, bottom: 0.82, square: 4 },
  ];
  const axles = [-3.05, 1.5, 2.85];
  const arches = axles.slice(0, 1).map((z) => ({ z, y: 0.5, radius: 0.6 }));
  const windshield = new RoundedBoxGeometry(width * 0.86, 0.95, 0.12, 3, 0.05);
  windshield.rotateX(-0.2);
  windshield.translate(0, 2.2, cabZ - 0.88);
  const sideWindows = [-1, 1].map((side) =>
    roundedBox(0.05, 0.72, 1.05, side * (half + 0.002), 2.2, cabZ - 0.2, 0.024),
  );
  return merged({
    body: [
      loft(cab, { rings: 48, segments: 44, deform: archDeform(arches, 0.26) }),
      roundedBox(width, 2.9, 5.7, 0, 2.15, 1.05, 0.11),
      roundedBox(width * 0.86, 0.34, 7.6, 0, 0.72, 0.1, 0.06),
    ],
    glass: [windshield, ...sideWindows],
    tyre: axles.flatMap((z) =>
      [-1, 1].map((side) => wheel(0.5, 0.42, side * (half - 0.22), 0.5, z)),
    ),
  });
}

function busParts(): ModelParts {
  const width = 2.54;
  const half = width / 2;
  const length = 12;
  const end = length / 2;
  const body: Station[] = [
    { z: end + 0.01, halfWidth: half * 0.86, top: 3.02, bottom: 0.46, square: 6 },
    { z: end - 0.08, halfWidth: half * 0.98, top: 3.16, bottom: 0.38, square: 7 },
    { z: end - 0.5, halfWidth: half, top: 3.2, bottom: 0.34, square: 8 },
    { z: 0, halfWidth: half, top: 3.2, bottom: 0.34, square: 8 },
    { z: -end + 0.5, halfWidth: half, top: 3.2, bottom: 0.34, square: 8 },
    { z: -end + 0.12, halfWidth: half * 0.98, top: 3.12, bottom: 0.38, square: 7 },
    { z: -end - 0.01, halfWidth: half * 0.86, top: 2.96, bottom: 0.46, square: 6 },
  ];
  const axles = [-3.3, 3.5];
  const arches = axles.map((z) => ({ z, y: 0.5, radius: 0.62 }));
  const windowY = 2.1;
  const windowHeight = 0.98;
  const sideBand = [-1, 1].map((side) =>
    roundedBox(0.04, windowHeight, length - 1.4, side * (half + 0.004), windowY, 0.1, 0.018),
  );
  const screen = roundedBox(width * 0.86, 1.5, 0.05, 0, 1.92, -end - 0.002, 0.024);
  const rear = roundedBox(width * 0.7, 0.72, 0.05, 0, 2.3, end + 0.002, 0.024);
  return merged({
    body: [loft(body, { rings: 90, segments: 56, deform: archDeform(arches, 0.26) })],
    glass: [...sideBand, screen, rear],
    tyre: axles.flatMap((z) =>
      [-1, 1].map((side) => wheel(0.5, 0.34, side * (half - 0.2), 0.5, z)),
    ),
  });
}

function thinWheel(radius: number, tube: number, z: number): BufferGeometry {
  const geometry = new TorusGeometry(radius - tube, tube, 12, 44);
  geometry.rotateY(Math.PI / 2);
  geometry.translate(0, radius, z);
  return geometry;
}

function spokes(radius: number, z: number): BufferGeometry {
  const geometry = new CylinderGeometry(radius * 0.86, radius * 0.86, 0.008, 28);
  geometry.rotateZ(Math.PI / 2);
  geometry.translate(0, radius, z);
  return geometry;
}

interface RiderPose {
  hip: Vector3;
  shoulder: Vector3;
  hand: Vector3;
  knee: Vector3;
  foot: Vector3;
  head: number;
}

function rider(pose: RiderPose): BufferGeometry[] {
  const { hip, shoulder, hand, knee, foot, head } = pose;
  const elbow = shoulder.clone().lerp(hand, 0.5).add(new Vector3(0, -0.08, 0.03));
  const parts: BufferGeometry[] = [limb(hip, shoulder, 0.17)];
  for (const side of [-1, 1]) {
    const at = (point: Vector3, spread: number): Vector3 =>
      point.clone().add(new Vector3(side * spread, 0, 0));
    parts.push(limb(at(hip, 0.1), at(knee, 0.12), 0.085));
    parts.push(limb(at(knee, 0.12), at(foot, 0.13), 0.065));
    parts.push(limb(at(shoulder, 0.2), at(elbow, 0.24), 0.056));
    parts.push(limb(at(elbow, 0.24), at(hand, 0.22), 0.047));
  }
  const neck = shoulder.clone().add(new Vector3(0, 0.15, -0.03));
  parts.push(limb(shoulder, neck, 0.05));
  parts.push(ball(head, neck.x, neck.y + head * 0.88, neck.z - 0.01));
  return parts;
}

function motorcycleParts(): ModelParts {
  const radius = 0.32;
  const frontZ = -0.72;
  const rearZ = 0.72;
  const hull: Station[] = [
    { z: 0.96, halfWidth: 0.05, top: 0.82, bottom: 0.72, square: 2.4 },
    { z: 0.7, halfWidth: 0.11, top: 0.85, bottom: 0.68, square: 2.6 },
    { z: 0.34, halfWidth: 0.15, top: 0.79, bottom: 0.55, square: 2.6 },
    { z: 0.04, halfWidth: 0.17, top: 0.86, bottom: 0.38, square: 2.6 },
    { z: -0.26, halfWidth: 0.2, top: 1.0, bottom: 0.4, square: 2.4 },
    { z: -0.46, halfWidth: 0.16, top: 1.0, bottom: 0.56, square: 2.4 },
    { z: -0.62, halfWidth: 0.05, top: 0.98, bottom: 0.82, square: 2.2 },
  ];
  const headlight = ball(0.09, 0, 0.98, -0.66);
  const fork = limb(new Vector3(0, radius, frontZ), new Vector3(0, 1.0, -0.56), 0.03);
  const swingarm = limb(new Vector3(0, radius, rearZ), new Vector3(0, 0.44, 0.1), 0.035);
  const bar = limb(new Vector3(-0.33, 1.06, -0.48), new Vector3(0.33, 1.06, -0.48), 0.018);
  const pose: RiderPose = {
    hip: new Vector3(0, 0.92, 0.4),
    shoulder: new Vector3(0, 1.44, 0.08),
    hand: new Vector3(0, 1.06, -0.46),
    knee: new Vector3(0, 0.74, -0.04),
    foot: new Vector3(0, 0.3, 0.22),
    head: 0.135,
  };
  return merged({
    body: [loft(hull, { rings: 44, segments: 24 }), headlight, ...rider(pose)],
    glass: [fork, swingarm, bar],
    tyre: [thinWheel(radius, 0.06, frontZ), thinWheel(radius, 0.065, rearZ)],
  });
}

function bicycleParts(): ModelParts {
  const radius = 0.34;
  const frontZ = -0.54;
  const rearZ = 0.54;
  const bottom = new Vector3(0, 0.3, 0.04);
  const seat = new Vector3(0, 0.78, 0.2);
  const head = new Vector3(0, 0.82, -0.36);
  const axleRear = new Vector3(0, radius, rearZ);
  const axleFront = new Vector3(0, radius, frontZ);
  const frame = [
    limb(bottom, seat, 0.018),
    limb(seat, head, 0.016),
    limb(bottom, head, 0.02),
    limb(bottom, axleRear, 0.012),
    limb(seat, axleRear, 0.012),
    limb(head, axleFront, 0.016),
    limb(head, new Vector3(0, 0.96, -0.32), 0.015),
    limb(new Vector3(-0.24, 0.96, -0.32), new Vector3(0.24, 0.96, -0.32), 0.013),
    limb(seat, new Vector3(0, 0.88, 0.22), 0.014),
  ];
  const saddle = new SphereGeometry(0.075, 14, 10);
  saddle.scale(0.8, 0.34, 1.7);
  saddle.translate(0, 0.9, 0.2);
  const pose: RiderPose = {
    hip: new Vector3(0, 0.96, 0.2),
    shoulder: new Vector3(0, 1.46, -0.03),
    hand: new Vector3(0, 0.98, -0.32),
    knee: new Vector3(0, 0.68, -0.14),
    foot: new Vector3(0, 0.26, 0.04),
    head: 0.125,
  };
  return merged({
    body: [saddle, ...rider(pose)],
    glass: [...frame, spokes(radius, frontZ), spokes(radius, rearZ)],
    tyre: [thinWheel(radius, 0.022, frontZ), thinWheel(radius, 0.022, rearZ)],
  });
}

function personParts(): ModelParts {
  const torso: Station[] = [
    { z: 0.0, halfWidth: 0.01, top: 0.002, bottom: -0.002, square: 2 },
    { z: 0.02, halfWidth: 0.1, top: 0.09, bottom: -0.09, square: 2.2 },
    { z: 0.18, halfWidth: 0.17, top: 0.1, bottom: -0.1, square: 2.2 },
    { z: 0.34, halfWidth: 0.15, top: 0.09, bottom: -0.09, square: 2.2 },
    { z: 0.52, halfWidth: 0.2, top: 0.1, bottom: -0.1, square: 2.4 },
    { z: 0.6, halfWidth: 0.19, top: 0.09, bottom: -0.09, square: 2.4 },
    { z: 0.645, halfWidth: 0.1, top: 0.05, bottom: -0.05, square: 2.2 },
    { z: 0.66, halfWidth: 0.01, top: 0.002, bottom: -0.002, square: 2 },
  ];
  const geometry = loft(torso, { rings: 44, segments: 24 });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, 0.84, 0);
  const shoulderY = 1.42;
  const hipY = 0.88;
  const parts: BufferGeometry[] = [geometry];
  for (const side of [-1, 1]) {
    parts.push(limb(new Vector3(side * 0.1, hipY, 0), new Vector3(side * 0.11, 0.48, -0.015), 0.078));
    parts.push(limb(new Vector3(side * 0.11, 0.48, -0.015), new Vector3(side * 0.12, 0.08, 0.02), 0.058));
    const foot = new SphereGeometry(0.06, 14, 10);
    foot.scale(0.85, 0.5, 1.8);
    foot.translate(side * 0.12, 0.04, -0.035);
    parts.push(foot);
    parts.push(
      limb(new Vector3(side * 0.205, shoulderY, 0), new Vector3(side * 0.24, 1.14, 0.01), 0.05),
    );
    parts.push(
      limb(new Vector3(side * 0.24, 1.14, 0.01), new Vector3(side * 0.245, 0.88, -0.02), 0.04),
    );
  }
  parts.push(limb(new Vector3(0, shoulderY + 0.04, 0), new Vector3(0, 1.56, -0.005), 0.05));
  const head = new SphereGeometry(0.105, 24, 16);
  head.scale(0.92, 1.1, 1);
  head.translate(0, 1.65, -0.01);
  parts.push(head);
  return merged({ body: parts });
}

let cache: Map<string, ModelParts> | null = null;

function library(): Map<string, ModelParts> {
  if (cache) return cache;
  cache = new Map<string, ModelParts>([
    ["sedan", carParts(SEDAN)],
    ["hatch", carParts(HATCH)],
    ["suv", carParts(SUV)],
    ["truck", truckParts()],
    ["bus", busParts()],
    ["motorcycle", motorcycleParts()],
    ["bicycle", bicycleParts()],
    ["person", personParts()],
  ]);
  return cache;
}

const CAR_VARIANTS = ["sedan", "hatch", "sedan", "suv"] as const;

/** Shape for an object. Same id, same car, for as long as it is on screen. */
export function modelName(className: ObjectClass, id: number): string {
  if (className === "car") return CAR_VARIANTS[id % CAR_VARIANTS.length]!;
  return className;
}

export function modelParts(name: string): ModelParts {
  const parts = library().get(name);
  if (!parts) throw new Error(`no model called ${name}`);
  return parts;
}
