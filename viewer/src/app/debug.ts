import { Vector2 } from "three";
import type { ObjectSample, SceneSegment, Timeline } from "../data/scene";
import type { World } from "../world/world";

const POINT = new Vector2();
const BOX_COLOR = "#7fe0a0";
const LOW_CONFIDENCE_COLOR = "#e6b295";

/** Distance labels on the clay, boxes on the video, and the camera numbers in a corner. */
export class DebugOverlay {
  enabled = false;
  private readonly pool: HTMLDivElement[] = [];

  constructor(
    private readonly labels: HTMLElement,
    private readonly panel: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly video: HTMLVideoElement,
  ) {}

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.panel.hidden = !enabled;
    this.canvas.hidden = !enabled;
    if (!enabled) {
      for (const label of this.pool) label.hidden = true;
    }
  }

  draw(world: World, samples: readonly ObjectSample[], timeline: Timeline, segment: number): void {
    if (!this.enabled) return;
    this.drawLabels(world, samples);
    this.drawVideo(samples, timeline.scene.segments[segment]);
  }

  private drawLabels(world: World, samples: readonly ObjectSample[]): void {
    let used = 0;
    for (const sample of samples) {
      if (sample.fade < 0.3 || !world.labelPosition(sample, POINT)) continue;
      const label = this.label(used++);
      const range = Math.hypot(sample.x, sample.y).toFixed(1);
      const flags = sample.flags.length ? `<br><i>${sample.flags.join(" ")}</i>` : "";
      label.innerHTML = `<b>${range} m</b> ${sample.className} ${sample.id} · ${sample.confidence.toFixed(2)}${flags}`;
      label.style.transform = `translate(${POINT.x}px, ${POINT.y}px) translate(-50%, -100%)`;
      label.hidden = false;
    }
    for (let index = used; index < this.pool.length; index++) this.pool[index]!.hidden = true;
  }

  private label(index: number): HTMLDivElement {
    let label = this.pool[index];
    if (!label) {
      label = document.createElement("div");
      label.className = "label";
      this.labels.append(label);
      this.pool.push(label);
    }
    return label;
  }

  /** Boxes in the video's own coordinates, inside its letterboxed area. */
  private drawVideo(samples: readonly ObjectSample[], segment: SceneSegment | undefined): void {
    const { canvas, video } = this;
    const ratio = Math.min(window.devicePixelRatio, 2);
    const paneWidth = canvas.clientWidth;
    const paneHeight = canvas.clientHeight;
    if (canvas.width !== Math.round(paneWidth * ratio)) canvas.width = Math.round(paneWidth * ratio);
    if (canvas.height !== Math.round(paneHeight * ratio)) {
      canvas.height = Math.round(paneHeight * ratio);
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, paneWidth, paneHeight);
    if (!video.videoWidth) return;

    const scale = Math.min(paneWidth / video.videoWidth, paneHeight / video.videoHeight);
    const width = video.videoWidth * scale;
    const height = video.videoHeight * scale;
    const left = (paneWidth - width) / 2;
    const top = (paneHeight - height) / 2;

    if (segment) {
      const { horizonFraction, visibleBottom } = segment.camera;
      context.strokeStyle = "rgb(255 220 120 / 80%)";
      context.lineWidth = 1;
      context.setLineDash([5, 5]);
      this.rule(context, left, top + horizonFraction * height, width);
      context.strokeStyle = "rgb(255 150 220 / 60%)";
      this.rule(context, left, top + (visibleBottom / video.videoHeight) * height, width);
      context.setLineDash([]);
    }

    context.font = "10.5px 'Geist Mono Variable', ui-monospace, monospace";
    for (const sample of samples) {
      if (!sample.box) continue;
      const [x1, y1, x2, y2] = sample.box;
      const low = sample.flags.includes("low_conf");
      context.strokeStyle = low ? LOW_CONFIDENCE_COLOR : BOX_COLOR;
      context.lineWidth = 1.25;
      const boxLeft = left + x1 * width;
      const boxTop = top + y1 * height;
      context.strokeRect(boxLeft, boxTop, (x2 - x1) * width, (y2 - y1) * height);
      const text = `${sample.id} ${Math.hypot(sample.x, sample.y).toFixed(0)}m`;
      const textWidth = context.measureText(text).width;
      context.fillStyle = "rgb(20 18 16 / 78%)";
      context.fillRect(boxLeft, boxTop - 14, textWidth + 8, 14);
      context.fillStyle = "#f8f6f1";
      context.fillText(text, boxLeft + 4, boxTop - 3.5);
    }
  }

  private rule(context: CanvasRenderingContext2D, left: number, y: number, width: number): void {
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(left + width, y);
    context.stroke();
  }

  /** The calibration behind the current shot, so a bad distance can be explained. */
  describe(timeline: Timeline, segment: number): void {
    const shot = timeline.scene.segments[segment];
    if (!shot) return;
    const { camera } = shot;
    const { egoMotion, model } = timeline.scene.metadata;
    const rows: [string, string][] = [
      ["shot", `${segment + 1} of ${timeline.scene.segments.length}`],
      ["horizon", `${camera.horizonFraction.toFixed(3)} · ${camera.horizonSource}`],
      ["height", `${camera.heightM.toFixed(2)} m · ${camera.heightSource}`],
      ["vfov", `${camera.vfovDeg.toFixed(0)}° · f ${camera.focalPx.toFixed(0)} px`],
      ["ego", `${egoMotion.mode} · ${(egoMotion.coverage * 100).toFixed(0)}%`],
      ["model", String(model.weights ?? "?")],
    ];
    this.panel.replaceChildren(
      ...rows.flatMap(([name, value]) => {
        const term = document.createElement("dt");
        term.textContent = name;
        const detail = document.createElement("dd");
        detail.textContent = value;
        return [term, detail];
      }),
    );
  }
}
