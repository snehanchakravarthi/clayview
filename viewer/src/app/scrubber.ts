import type { Timeline } from "../data/scene";

const BAR_STEP = 3;
const BAR_WIDTH = 2;
const CLAY = "#c4704a";
const INK = "#2f2b26";
const REMAINING = "rgb(47 43 38 / 20%)";
const PLAYED = "rgb(47 43 38 / 72%)";

/** A timeline you can read: bar height is how much traffic is on screen, ticks are cuts. */
export class Scrubber {
  private bins = new Float32Array(0);
  private timeline: Timeline | null = null;
  private width = 0;
  private height = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    new ResizeObserver(() => this.measure()).observe(canvas);
  }

  setTimeline(timeline: Timeline | null): void {
    this.timeline = timeline;
    this.measure();
  }

  private measure(): void {
    const box = this.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio, 2);
    this.width = box.width;
    this.height = box.height;
    this.canvas.width = Math.round(box.width * ratio);
    this.canvas.height = Math.round(box.height * ratio);
    this.canvas.getContext("2d")?.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.computeBins();
  }

  private computeBins(): void {
    const timeline = this.timeline;
    const count = Math.max(Math.floor(this.width / BAR_STEP), 1);
    this.bins = new Float32Array(count);
    if (!timeline) return;
    let peak = 1;
    for (let index = 0; index < timeline.times.length; index++) {
      const bin = Math.min(Math.floor((timeline.times[index]! / timeline.duration) * count), count - 1);
      this.bins[bin] = Math.max(this.bins[bin]!, timeline.density[index]!);
      peak = Math.max(peak, timeline.density[index]!);
    }
    for (let bin = 0; bin < count; bin++) this.bins[bin] = this.bins[bin]! / peak;
  }

  draw(time: number): void {
    const context = this.canvas.getContext("2d");
    const timeline = this.timeline;
    if (!context || !timeline || timeline.duration <= 0) return;
    const { width, height } = this;
    context.clearRect(0, 0, width, height);
    const progress = time / timeline.duration;
    const baseline = height - 8;
    const room = baseline - 6;

    for (let bin = 0; bin < this.bins.length; bin++) {
      const barHeight = Math.max(2, this.bins[bin]! * room);
      const x = bin * BAR_STEP;
      context.fillStyle = x / width <= progress ? PLAYED : REMAINING;
      context.beginPath();
      context.roundRect(x, baseline - barHeight, BAR_WIDTH, barHeight, 1);
      context.fill();
    }

    context.fillStyle = CLAY;
    for (const cut of timeline.cuts) {
      const x = (cut / timeline.duration) * width;
      context.fillRect(x - 0.75, 3, 1.5, height - 6);
    }

    const x = Math.max(1, Math.min(width - 1, progress * width));
    context.fillStyle = INK;
    context.beginPath();
    context.roundRect(x - 1, 1, 2, height - 2, 1);
    context.fill();
  }
}
