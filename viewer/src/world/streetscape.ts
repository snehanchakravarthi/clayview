import { Group, type Mesh, MeshStandardMaterial } from "three";
import { cleanMarkings } from "../data/paint";
import {
  egoFrameTransform,
  type EgoPose,
  type SceneFile,
  type WorldSegment,
} from "../data/scene";
import { buildStreetLine, densify, StreetIndex } from "../data/street";
import { withCutaway } from "./cutaway";
import { Flora } from "./flora";
import { layMassing } from "./massing";
import { layBarriers, layFurniture } from "./props";
import { Soup } from "./soup";
import { layIslands, layPaint, laySurface } from "./surface";

/** Past this from ego a patch of the world is lost in the fog, so don't draw it. */
const DRAW_RANGE = 270;
const MASSING_FOG_SCALE = 1.28;

export interface StreetCounts {
  roadM: number;
  paint: number;
  trees: number;
  bushes: number;
  furniture: number;
  islands: number;
  barriers: number;
  blocks: number;
  triangles: number;
  meshes: number;
  buildMs: number;
}

const surfaceMaterial = new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 });
const massingMaterial = withCutaway(
  new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }),
);
massingMaterial.defines = { ...massingMaterial.defines, FOG_SCALE: MASSING_FOG_SCALE.toFixed(2) };

/** The built street of one shot, in that shot's own world frame. */
class Shot {
  readonly group = new Group();
  readonly flora: Flora;
  readonly counts: StreetCounts;
  private readonly patches: Mesh[];

  constructor(world: WorldSegment) {
    const started = performance.now();
    const line = buildStreetLine(world.road);
    if (!line) throw new Error("a shot's world needs at least two road stations");
    const index = new StreetIndex(line);

    const ground = new Soup();
    laySurface(ground, densify(line));
    const paint = layPaint(ground, cleanMarkings(world.markings, index));
    const islands = layIslands(ground, world.objects, index);

    const props = new Soup();
    const furniture = layFurniture(props, world.objects, index);
    const barriers = layBarriers(props, world.barriers, index);

    const massing = new Soup();
    const blocks = layMassing(massing, world.buildings, index);

    const triangles = ground.triangles + props.triangles + massing.triangles;
    this.patches = [
      ...ground.meshes(surfaceMaterial, { cast: false, receive: true }),
      ...props.meshes(surfaceMaterial, { cast: true, receive: true }),
      ...massing.meshes(massingMaterial, { cast: true, receive: true }),
    ];
    this.flora = new Flora(world.objects);
    this.group.add(...this.patches, this.flora.group);
    const stations = world.road;
    this.counts = {
      roadM: stations[stations.length - 1]!.s - stations[0]!.s,
      paint,
      trees: this.flora.treeCount,
      bushes: this.flora.bushCount,
      furniture,
      islands,
      barriers,
      blocks,
      triangles,
      meshes: this.patches.length,
      buildMs: performance.now() - started,
    };
  }

  /** Hide what's past the fog. Frustum culling takes care of the rest. */
  update(pose: EgoPose): void {
    const x = pose.x;
    const z = -pose.y;
    for (const mesh of this.patches) {
      const sphere = mesh.geometry.boundingSphere;
      if (!sphere) continue;
      const range = Math.hypot(sphere.center.x - x, sphere.center.z - z) - sphere.radius;
      mesh.visible = range < DRAW_RANGE;
    }
    this.flora.update(pose.x, pose.y);
  }

  dispose(): void {
    for (const mesh of this.patches) mesh.geometry.dispose();
    this.flora.dispose();
  }
}

/** The real street of every shot in a scene. Static: ego moves through it, it never changes. */
export class Streetscape {
  /** Sits in the ego frame. Its children are in world metres; this group carries them. */
  readonly group = new Group();
  private readonly shots = new Map<number, Shot>();
  private showing: Shot | null = null;

  /** Build everything up front, so a cut can swap worlds without a hitch. */
  load(scene: SceneFile | null): void {
    for (const shot of this.shots.values()) shot.dispose();
    this.shots.clear();
    this.group.clear();
    this.showing = null;
    for (const world of scene?.world?.segments ?? []) {
      if (world.road.length < 2) continue;
      const shot = new Shot(world);
      shot.group.visible = false;
      this.shots.set(world.segment, shot);
      this.group.add(shot.group);
    }
  }

  /** Show a shot's world at ego's pose. False when that shot has none. */
  update(segment: number, pose: EgoPose): boolean {
    const shot = this.shots.get(segment) ?? null;
    if (shot !== this.showing) {
      if (this.showing) this.showing.group.visible = false;
      if (shot) shot.group.visible = true;
      this.showing = shot;
    }
    if (!shot) return false;
    const { x, z, rotationY } = egoFrameTransform(pose);
    this.group.position.set(x, 0, z);
    this.group.rotation.y = rotationY;
    shot.update(pose);
    return true;
  }

  get counts(): StreetCounts | null {
    return this.showing?.counts ?? null;
  }
}
