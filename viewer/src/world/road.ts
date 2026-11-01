import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  DoubleSide,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  Quaternion,
  RingGeometry,
  Shape,
  ShapeGeometry,
  SRGBColorSpace,
  Vector3,
} from "three";
import { PALETTE } from "./palette";

const LANE_WIDTH = 3.5;
const LANE_COUNT = 3;
const ROAD_HALF_WIDTH = (LANE_WIDTH * LANE_COUNT) / 2;
const START = -46;
const END = 236;
const SURFACE_STEPS = 56;
const DASH_LENGTH = 3;
const DASH_PERIOD = 9;
const DASH_WIDTH = 0.13;
const EDGE_WIDTH = 0.12;
const EDGE_PIECE = 2.4;
const RINGS = [10, 25, 50, 100] as const;

const MATRIX = new Matrix4();
const POSITION = new Vector3();
const ROTATION = new Quaternion();
const SCALE = new Vector3(1, 1, 1);
const UP = new Vector3(0, 1, 0);

function roundedDash(length: number, width: number): ShapeGeometry {
  const halfLength = length / 2;
  const radius = width / 2;
  const shape = new Shape();
  shape.moveTo(-radius, -halfLength + radius);
  shape.absarc(0, -halfLength + radius, radius, Math.PI, 0, false);
  shape.lineTo(radius, halfLength - radius);
  shape.absarc(0, halfLength - radius, radius, 0, Math.PI, false);
  shape.closePath();
  const geometry = new ShapeGeometry(shape, 6);
  geometry.rotateX(Math.PI / 2);
  return geometry;
}

function lateralOnArc(offset: number, distance: number, curvature: number): number {
  return offset + 0.5 * curvature * distance * distance;
}

/**
 * The road is decoration, not perception: nothing detects lanes. It carries the sense of motion,
 * scrolling at the estimated ego speed and bending with the estimated yaw.
 */
export class Road {
  readonly group = new Group();
  /** The distance rings, on their own so they can sit on the real street too. */
  readonly rings: Group;
  private readonly surface: Mesh<BufferGeometry, MeshStandardMaterial>;
  private readonly dashes: InstancedMesh;
  private readonly edges: InstancedMesh;

  constructor() {
    const geometry = new BufferGeometry();
    const vertices = (SURFACE_STEPS + 1) * 2;
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(vertices * 3), 3));
    geometry.setAttribute("normal", new BufferAttribute(new Float32Array(vertices * 3), 3));
    const normals = geometry.getAttribute("normal") as BufferAttribute;
    for (let index = 0; index < vertices; index++) normals.setXYZ(index, 0, 1, 0);
    const indices: number[] = [];
    for (let step = 0; step < SURFACE_STEPS; step++) {
      const a = step * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    geometry.setIndex(indices);
    this.surface = new Mesh(
      geometry,
      new MeshStandardMaterial({ color: PALETTE.road, roughness: 1, metalness: 0 }),
    );
    this.surface.frustumCulled = false;
    this.surface.receiveShadow = true;
    this.surface.position.y = 0.01;

    const paint = new MeshStandardMaterial({
      color: PALETTE.marking,
      roughness: 0.9,
      metalness: 0,
      side: DoubleSide,
    });
    const dashCount = Math.ceil((END - START) / DASH_PERIOD) * (LANE_COUNT - 1);
    this.dashes = new InstancedMesh(roundedDash(DASH_LENGTH, DASH_WIDTH), paint, dashCount);
    this.dashes.frustumCulled = false;
    this.dashes.receiveShadow = true;
    const edgeCount = Math.ceil((END - START) / EDGE_PIECE) * 2;
    this.edges = new InstancedMesh(roundedDash(EDGE_PIECE * 1.2, EDGE_WIDTH), paint, edgeCount);
    this.edges.frustumCulled = false;
    this.edges.receiveShadow = true;

    this.rings = this.buildRings();
    this.group.add(this.surface, this.dashes, this.edges);
    this.update(0, 0);
  }

  private buildRings(): Group {
    const rings = new Group();
    for (const radius of RINGS) {
      // Far rings lie almost edge-on and would read as a horizon line at full strength.
      const material = new MeshBasicMaterial({
        color: PALETTE.ring,
        transparent: true,
        opacity: 0.4 * (1 - radius / 130),
        depthWrite: false,
      });
      const weight = 0.035 + radius * 0.0012;
      const ring = new Mesh(new RingGeometry(radius - weight, radius + weight, 192), material);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.03;
      rings.add(ring);

      const label = ringLabel(`${radius} m`);
      const scale = 0.42 + radius * 0.022;
      label.scale.set(scale * 2, scale, 1);
      const bearing = (Math.PI / 180) * 16;
      label.position.set(
        Math.sin(bearing) * (radius + scale * 0.7),
        0.04,
        -Math.cos(bearing) * (radius + scale * 0.7),
      );
      label.rotation.x = -Math.PI / 2;
      rings.add(label);
    }
    return rings;
  }

  /** Lay the road out for the current travelled distance and path curvature. */
  update(distance: number, curvature: number): void {
    this.layoutSurface(curvature);
    this.layoutMarkings(this.dashes, distance, curvature, DASH_PERIOD, [
      -LANE_WIDTH / 2,
      LANE_WIDTH / 2,
    ]);
    this.layoutMarkings(this.edges, distance, curvature, EDGE_PIECE, [
      -ROAD_HALF_WIDTH,
      ROAD_HALF_WIDTH,
    ]);
  }

  private layoutSurface(curvature: number): void {
    const positions = this.surface.geometry.getAttribute("position") as BufferAttribute;
    for (let step = 0; step <= SURFACE_STEPS; step++) {
      const distance = START + ((END - START) * step) / SURFACE_STEPS;
      const center = lateralOnArc(0, distance, curvature);
      positions.setXYZ(step * 2, center - ROAD_HALF_WIDTH, 0, -distance);
      positions.setXYZ(step * 2 + 1, center + ROAD_HALF_WIDTH, 0, -distance);
    }
    positions.needsUpdate = true;
  }

  private layoutMarkings(
    mesh: InstancedMesh,
    distance: number,
    curvature: number,
    period: number,
    offsets: readonly number[],
  ): void {
    const scroll = ((distance % period) + period) % period;
    const perLine = Math.floor(mesh.count / offsets.length);
    let index = 0;
    for (const offset of offsets) {
      for (let piece = 0; piece < perLine; piece++) {
        const along = START + piece * period - scroll;
        const yaw = -curvature * along;
        POSITION.set(lateralOnArc(offset, along, curvature), 0.022, -along);
        ROTATION.setFromAxisAngle(UP, yaw);
        MATRIX.compose(POSITION, ROTATION, SCALE);
        mesh.setMatrixAt(index++, MATRIX);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
}

function ringLabel(text: string): Mesh {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (context) {
    context.font = "500 64px 'Geist Mono Variable', ui-monospace, monospace";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#6f685c";
    context.fillText(text, 128, 64);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = 8;
  return new Mesh(
    new PlaneGeometry(1, 1),
    new MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.72, depthWrite: false }),
  );
}
