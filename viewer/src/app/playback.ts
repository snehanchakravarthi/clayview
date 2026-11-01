/** One clock for the whole app. With a video, the video is the clock; without one, this is. */
export class Playback {
  private video: HTMLVideoElement | null = null;
  private seconds = 0;
  private running = false;
  private lastNow = 0;
  private speed = 1;
  duration = 0;

  attach(video: HTMLVideoElement | null): void {
    this.video = video;
    if (!video) return;
    video.playbackRate = this.speed;
    video.currentTime = this.seconds;
    if (this.running) void video.play().catch(() => this.pause());
  }

  get rate(): number {
    return this.speed;
  }

  set rate(value: number) {
    this.speed = value;
    if (this.video) this.video.playbackRate = value;
  }

  get playing(): boolean {
    return this.video ? !this.video.paused && !this.video.ended : this.running;
  }

  get time(): number {
    return this.video ? this.video.currentTime : this.seconds;
  }

  play(): void {
    if (this.time >= this.duration - 0.02) this.seek(0);
    this.running = true;
    this.lastNow = performance.now();
    if (this.video) void this.video.play().catch(() => this.pause());
  }

  pause(): void {
    this.running = false;
    this.video?.pause();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(seconds: number): void {
    const clamped = Math.max(0, Math.min(seconds, this.duration));
    this.seconds = clamped;
    if (this.video) this.video.currentTime = clamped;
  }

  /** Call once per animation frame. */
  tick(now: number): void {
    if (this.video) {
      this.seconds = this.video.currentTime;
      return;
    }
    if (!this.running) return;
    this.seconds += ((now - this.lastNow) / 1000) * this.speed;
    this.lastNow = now;
    if (this.seconds >= this.duration) {
      this.seconds = this.duration;
      this.running = false;
    }
  }
}
