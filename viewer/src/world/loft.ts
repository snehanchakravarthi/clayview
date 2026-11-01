import { BufferGeometry, Float32BufferAttribute, Vector3 } from "three";

/** One cross-section of a lofted form. z runs along the length; nose towards -z. */
export interface Station {
  z: number;
  halfWidth: number;
  top: number;
  bottom: number;
  /** Superellipse exponent: 2 is an ellipse, higher is a rounded box. */
  square?: number;
  /** How much the sides lean in towards the top, as a fraction of the width. */
  lean?: number;
}

export interface LoftOptions {
  rings?: number;
  segments?: number;
  deform?: (vertex: Vector3) => void;
}

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (p2 - p0) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (3 * p1 - p0 - 3 * p2 + p3) * t3)
  );
}

function interpolate(stations: readonly Station[], position: number): Required<Station> {
  const last = stations.length - 1;
  const scaled = Math.max(0, Math.min(1, position)) * last;
  const index = Math.min(Math.floor(scaled), last - 1);
  const t = scaled - index;
  const a = stations[Math.max(index - 1, 0)]!;
  const b = stations[index]!;
  const c = stations[index + 1]!;
  const d = stations[Math.min(index + 2, last)]!;
  const blend = (read: (station: Station) => number): number =>
    catmullRom(read(a), read(b), read(c), read(d), t);
  return {
    z: blend((s) => s.z),
    halfWidth: Math.max(blend((s) => s.halfWidth), 0.001),
    top: blend((s) => s.top),
    bottom: blend((s) => s.bottom),
    square: blend((s) => s.square ?? 3),
    lean: blend((s) => s.lean ?? 0),
  };
}

function signedPower(value: number, exponent: number): number {
  return Math.sign(value) * Math.abs(value) ** exponent;
}

/**
 * Sweep a superellipse through stations with a Catmull-Rom spine, closing both ends.
 * Vertices are shared around each ring, so normals stay smooth over the seam.
 */
export function loft(stations: readonly Station[], options: LoftOptions = {}): BufferGeometry {
  const rings = options.rings ?? 56;
  const segments = options.segments ?? 40;
  const positions: number[] = [];
  const indices: number[] = [];
  const vertex = new Vector3();

  const firstRing = interpolate(stations, 0);
  const lastRing = interpolate(stations, 1);

  const push = (): void => {
    options.deform?.(vertex);
    positions.push(vertex.x, vertex.y, vertex.z);
  };

  vertex.set(0, (firstRing.top + firstRing.bottom) / 2, firstRing.z);
  push();
  for (let ring = 0; ring <= rings; ring++) {
    const section = interpolate(stations, ring / rings);
    const middle = (section.top + section.bottom) / 2;
    const radius = (section.top - section.bottom) / 2;
    const exponent = 2 / section.square;
    for (let segment = 0; segment < segments; segment++) {
      const angle = (segment / segments) * Math.PI * 2;
      const across = signedPower(Math.cos(angle), exponent);
      const up = signedPower(Math.sin(angle), exponent);
      const lean = 1 - section.lean * Math.max(up, 0);
      vertex.set(section.halfWidth * across * lean, middle + radius * up, section.z);
      push();
    }
  }
  vertex.set(0, (lastRing.top + lastRing.bottom) / 2, lastRing.z);
  push();

  const ringStart = (ring: number): number => 1 + ring * segments;
  // Stations run from tail to nose along -z, so this winding faces outward.
  for (let segment = 0; segment < segments; segment++) {
    const next = (segment + 1) % segments;
    indices.push(0, ringStart(0) + segment, ringStart(0) + next);
  }
  for (let ring = 0; ring < rings; ring++) {
    for (let segment = 0; segment < segments; segment++) {
      const next = (segment + 1) % segments;
      const a = ringStart(ring) + segment;
      const b = ringStart(ring) + next;
      const c = ringStart(ring + 1) + segment;
      const d = ringStart(ring + 1) + next;
      indices.push(a, c, b, b, c, d);
    }
  }
  const tip = positions.length / 3 - 1;
  for (let segment = 0; segment < segments; segment++) {
    const next = (segment + 1) % segments;
    indices.push(tip, ringStart(rings) + next, ringStart(rings) + segment);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}
