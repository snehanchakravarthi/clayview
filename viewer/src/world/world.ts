import { Group, type Vector2 } from "three";
import type { ObjectSample, Timeline } from "../data/scene";
import { sampleEgo, sampleTrack } from "../data/scene";
import { Actor, CLAY, Traffic } from "./actors";
import { Road } from "./road";
import { buildShowroom } from "./showroom";
import { Stage } from "./stage";
import { Streetscape, type StreetCounts } from "./streetscape";
import { ViewRig, type ViewName } from "./views";

/** Camera sits on the dash, a bit behind the nose, so the car model goes behind the origin. */
const EGO_OFFSET_Z = 0.55;
const TURNTABLE_RATE = 0.07;

/** What the last frame cost, for the debug overlay. CPU time only: the GPU runs after it. */
export interface FrameCost {
  renderMs: number;
  calls: number;
  triangles: number;
}

export interface WorldReadout {
  speed: number;
  objects: number;
  segment: number;
  samples: readonly ObjectSample[];
  /** The video camera's yaw on its mount, relative to the car. */
  mount: number;
}

/** Everything in the clay pane. The app gives it a time; it draws that instant. */
export class World {
  private readonly stage: Stage;
  private readonly road = new Road();
  private readonly street = new Streetscape();
  private readonly traffic = new Traffic();
  private readonly ego = new Actor("sedan", CLAY);
  private readonly showroom: Group;
  private readonly rig: ViewRig;
  private readonly samples: ObjectSample[] = [];
  private timeline: Timeline | null = null;
  private showRings = false;
  readonly lastFrame: FrameCost = { renderMs: 0, calls: 0, triangles: 0 };

  constructor(canvas: HTMLCanvasElement, onFreeView: () => void) {
    this.stage = new Stage(canvas);
    this.rig = new ViewRig(this.stage.camera, canvas, onFreeView);
    this.ego.setOpacity(1);
    this.ego.group.position.z = EGO_OFFSET_Z;
    this.showroom = buildShowroom();
    this.road.group.visible = false;
    this.road.rings.visible = false;
    this.stage.scene.add(
      this.road.group,
      this.road.rings,
      this.street.group,
      this.traffic.group,
      this.ego.group,
      this.showroom,
    );
    this.rig.showroom();
  }

  /** Range rings over the real street. The decorative road always has them. */
  setRings(show: boolean): void {
    this.showRings = show;
  }

  get view(): ViewName | "free" {
    return this.rig.current;
  }

  setView(name: ViewName): void {
    this.rig.go(name);
  }

  /** Swap between the empty-state turntable and a loaded scene. */
  setTimeline(timeline: Timeline | null): void {
    const changed = (timeline === null) !== (this.timeline === null);
    this.timeline = timeline;
    this.showroom.visible = timeline === null;
    this.road.group.visible = timeline !== null;
    this.street.load(timeline?.scene ?? null);
    this.ego.group.position.x = 0;
    if (changed) {
      if (timeline) this.rig.go("chase", true);
      else this.rig.showroom();
    }
    if (!timeline) {
      this.traffic.update([]);
      this.road.rings.visible = false;
    }
  }

  /** Draw time t. Returns what the UI needs to show alongside it. */
  frame(t: number, dt: number): WorldReadout | null {
    const timeline = this.timeline;
    let readout: WorldReadout | null = null;
    let lean = 0;
    if (timeline) {
      this.samples.length = 0;
      for (const track of timeline.tracks) {
        if (t < track.start || t > track.end) continue;
        const sample = sampleTrack(track, t);
        if (sample) this.samples.push(sample);
      }
      this.traffic.update(this.samples);
      const ego = sampleEgo(timeline, t);
      const onStreet = this.street.update(ego.segment, ego.pose);
      this.road.group.visible = !onStreet;
      this.road.rings.visible = !onStreet || this.showRings;
      this.road.setRingsOnStreet(onStreet);
      if (onStreet) {
        this.stage.aimSun(ego.pose.x, ego.pose.y, ego.pose.yaw);
        lean = ego.pose.yaw - ego.easedYaw;
      } else {
        this.road.update(ego.distance, ego.curvature);
        this.stage.aimSun(0, 0, 0);
      }
      readout = {
        speed: ego.speed,
        objects: this.samples.length,
        segment: ego.segment,
        samples: this.samples,
        mount: ego.mount,
      };
    } else if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.orbit(dt);
    }
    this.rig.update(dt);
    this.rig.lean(lean);
    const started = performance.now();
    this.stage.render();
    this.lastFrame.renderMs = performance.now() - started;
    this.rig.settle();
    this.lastFrame.calls = this.stage.drawn.calls;
    this.lastFrame.triangles = this.stage.drawn.triangles;
    return readout;
  }

  private orbit(dt: number): void {
    const controls = this.rig.controls;
    controls.autoRotate = false;
    const camera = this.stage.camera;
    const angle = TURNTABLE_RATE * dt;
    const dx = camera.position.x - controls.target.x;
    const dz = camera.position.z - controls.target.z;
    camera.position.x = controls.target.x + dx * Math.cos(angle) - dz * Math.sin(angle);
    camera.position.z = controls.target.z + dx * Math.sin(angle) + dz * Math.cos(angle);
  }

  /** What the current shot's street is made of, or null when it's the decorative road. */
  get streetCounts(): StreetCounts | null {
    return this.street.counts;
  }

  labelPosition(sample: ObjectSample, out: Vector2): boolean {
    return this.stage.project(sample.x, 2.3, -sample.y, out);
  }
}
