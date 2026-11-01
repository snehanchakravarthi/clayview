import { Vector3, type PerspectiveCamera } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type ViewName = "chase" | "aerial" | "top";

interface View {
  position: Vector3;
  target: Vector3;
}

const VIEWS: Record<ViewName, View> = {
  chase: { position: new Vector3(0, 6.6, 21), target: new Vector3(0, 0.4, -15) },
  aerial: { position: new Vector3(-30, 44, 42), target: new Vector3(0, 0, -20) },
  top: { position: new Vector3(0, 122, -26.01), target: new Vector3(0, 0, -26) },
};

const SHOWROOM: View = {
  position: new Vector3(13, 5.4, 22),
  target: new Vector3(0, 0.75, 1),
};

const TRANSITION_SECONDS = 0.9;

function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

/** Orbit camera plus named presets it glides between. Dragging leaves the preset. */
export class ViewRig {
  readonly controls: OrbitControls;
  current: ViewName | "free" = "chase";
  private fromPosition = new Vector3();
  private fromTarget = new Vector3();
  private toPosition = new Vector3();
  private toTarget = new Vector3();
  private progress = 1;

  constructor(
    private readonly camera: PerspectiveCamera,
    element: HTMLElement,
    private readonly onFree: () => void,
  ) {
    this.controls = new OrbitControls(camera, element);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 7;
    this.controls.maxDistance = 240;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.035;
    this.controls.addEventListener("start", () => {
      this.progress = 1;
      this.current = "free";
      this.onFree();
    });
    this.jump(VIEWS.chase);
  }

  private jump(view: View): void {
    this.camera.position.copy(view.position);
    this.controls.target.copy(view.target);
    this.progress = 1;
    this.controls.update();
  }

  showroom(): void {
    this.jump(SHOWROOM);
    this.current = "free";
  }

  go(name: ViewName, instant = false): void {
    this.current = name;
    const view = VIEWS[name];
    if (instant || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.jump(view);
      return;
    }
    this.fromPosition.copy(this.camera.position);
    this.fromTarget.copy(this.controls.target);
    this.toPosition.copy(view.position);
    this.toTarget.copy(view.target);
    this.progress = 0;
  }

  update(dt: number): void {
    if (this.progress < 1) {
      this.progress = Math.min(this.progress + dt / TRANSITION_SECONDS, 1);
      const mix = easeInOut(this.progress);
      this.camera.position.lerpVectors(this.fromPosition, this.toPosition, mix);
      this.controls.target.lerpVectors(this.fromTarget, this.toTarget, mix);
    }
    this.controls.update();
  }
}
