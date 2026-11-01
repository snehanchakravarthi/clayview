import { Group, Mesh, MeshStandardMaterial, type Color } from "three";
import type { ObjectClass, ObjectSample } from "../data/scene";
import { modelName, modelParts, type Tone } from "./models";
import { PALETTE } from "./palette";

const ENTER_SCALE = 0.88;
const FAR_FADE_START = 70;
const FAR_FADE_END = 125;
const FAR_OPACITY = 0.35;
const LOW_CONFIDENCE = 0.22;
const FULL_CONFIDENCE = 0.55;
const LOW_CONFIDENCE_OPACITY = 0.4;

interface ToneSet {
  body: Color;
  glass: Color;
  tyre: Color;
}

export const FOAM: ToneSet = {
  body: PALETTE.foam,
  glass: PALETTE.foamGlass,
  tyre: PALETTE.foamTyre,
};

export const CLAY: ToneSet = {
  body: PALETTE.clay,
  glass: PALETTE.clayGlass,
  tyre: PALETTE.clayTyre,
};

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** Opacity for an object: its fade envelope, dimmed when it is far or barely believed. */
export function objectOpacity(sample: ObjectSample, range: number): number {
  const trust =
    LOW_CONFIDENCE_OPACITY +
    (1 - LOW_CONFIDENCE_OPACITY) * smoothstep(LOW_CONFIDENCE, FULL_CONFIDENCE, sample.confidence);
  const distance = 1 - (1 - FAR_OPACITY) * smoothstep(FAR_FADE_START, FAR_FADE_END, range);
  return sample.fade * trust * distance;
}

/** A model with its own materials, so each one can fade alone. */
export class Actor {
  readonly group = new Group();
  private readonly materials: MeshStandardMaterial[] = [];

  constructor(readonly model: string, tones: ToneSet) {
    const parts = modelParts(model);
    for (const tone of Object.keys(parts) as Tone[]) {
      const geometry = parts[tone];
      if (!geometry) continue;
      const material = new MeshStandardMaterial({
        color: tones[tone],
        roughness: tone === "glass" ? 0.62 : 0.86,
        metalness: 0,
      });
      const mesh = new Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.materials.push(material);
      this.group.add(mesh);
    }
  }

  setOpacity(opacity: number): void {
    const solid = opacity >= 0.995;
    for (const material of this.materials) {
      material.opacity = opacity;
      material.transparent = !solid;
    }
  }
}

/** Keeps one actor per live object, and reuses them as objects come and go. */
export class Traffic {
  readonly group = new Group();
  private readonly live = new Map<number, Actor>();
  private readonly spare = new Map<string, Actor[]>();
  private readonly seen = new Set<number>();

  /** Place every sampled object for this instant. Rebuilt whole, so scrubbing just works. */
  update(samples: readonly ObjectSample[]): void {
    this.seen.clear();
    for (const sample of samples) {
      this.seen.add(sample.id);
      const actor = this.live.get(sample.id) ?? this.acquire(sample.className, sample.id);
      const range = Math.hypot(sample.x, sample.y);
      actor.group.position.set(sample.x, 0, -sample.y);
      actor.group.rotation.y = -sample.heading;
      actor.group.scale.setScalar(ENTER_SCALE + (1 - ENTER_SCALE) * sample.fade);
      actor.setOpacity(objectOpacity(sample, range));
    }
    for (const [id, actor] of this.live) {
      if (this.seen.has(id)) continue;
      this.live.delete(id);
      actor.group.visible = false;
      this.spare.get(actor.model)?.push(actor);
    }
  }

  private acquire(className: ObjectClass, id: number): Actor {
    const name = modelName(className, id);
    let bucket = this.spare.get(name);
    if (!bucket) {
      bucket = [];
      this.spare.set(name, bucket);
    }
    let actor = bucket.pop();
    if (!actor) {
      actor = new Actor(name, FOAM);
      this.group.add(actor.group);
    }
    actor.group.visible = true;
    this.live.set(id, actor);
    return actor;
  }
}
