import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "./styles.css";
import { DebugOverlay } from "./app/debug";
import { fetchScene, fileName, findLatest, readFiles, type VideoSource } from "./app/loader";
import { Playback } from "./app/playback";
import { Scrubber } from "./app/scrubber";
import { buildTimeline, SceneError, type SceneFile, type Timeline } from "./data/scene";
import type { ViewName } from "./world/views";
import { World } from "./world/world";

const SECOND_SKIP = 5;
const KMH = 3.6;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing #${id}`);
  return found as T;
}

const app = element<HTMLDivElement>("app");
const stage = element<HTMLElement>("stage");
const video = element<HTMLVideoElement>("video");
const videoName = element<HTMLElement>("video-name");
const videoMissing = element<HTMLElement>("video-missing");
const videoMissingText = element<HTMLElement>("video-missing-text");
const playButton = element<HTMLButtonElement>("play");
const scrub = element<HTMLInputElement>("scrub");
const timeNow = element<HTMLElement>("time-now");
const timeTotal = element<HTMLElement>("time-total");
const hudSpeed = element<HTMLElement>("hud-speed");
const hudMeta = element<HTMLElement>("hud-meta");
const views = element<HTMLElement>("views");
const rates = element<HTMLElement>("rates");
const debugToggle = element<HTMLButtonElement>("debug");
const picker = element<HTMLInputElement>("picker");
const veil = element<HTMLElement>("veil");
const toast = element<HTMLElement>("toast");
const openLatest = element<HTMLButtonElement>("open-latest");
const splitter = element<HTMLElement>("splitter");

const playback = new Playback();
const scrubber = new Scrubber(element<HTMLCanvasElement>("scrub-canvas"));
const world = new World(element<HTMLCanvasElement>("clay"), () => syncViewButtons());
const debug = new DebugOverlay(
  element("labels"),
  element("debug-panel"),
  element<HTMLCanvasElement>("video-debug"),
  video,
);

let timeline: Timeline | null = null;
let videoLoaded = false;
let objectUrl: string | null = null;
let scrubbing = false;
let lastSegment = -1;
let lastFrameTime = performance.now();
let toastTimer = 0;

function formatTime(seconds: number): string {
  const safe = Math.max(seconds, 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

function showToast(message: string): void {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 6000);
}

function showError(error: unknown): void {
  showToast(error instanceof SceneError ? error.message : "Something went wrong opening that.");
  if (!(error instanceof SceneError)) console.error(error);
}

function syncViewButtons(): void {
  const current = world.view;
  for (const button of views.querySelectorAll<HTMLButtonElement>("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === current));
  }
}

function setScene(scene: SceneFile): void {
  timeline = buildTimeline(scene);
  playback.pause();
  playback.duration = timeline.duration;
  playback.seek(0);
  world.setTimeline(timeline);
  scrubber.setTimeline(timeline);
  app.dataset.state = "ready";
  timeTotal.textContent = formatTime(timeline.duration);
  lastSegment = -1;
  document.title = `${fileName(scene.metadata.source.path)} · clayview`;
  syncViewButtons();
  syncVideoPane();
}

function setVideo(source: VideoSource): void {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = source.url.startsWith("blob:") ? source.url : null;
  videoLoaded = true;
  video.src = source.url;
  video.load();
  videoName.textContent = source.name;
  playback.attach(video);
  syncVideoPane();
}

function clearVideo(): void {
  videoLoaded = false;
  video.removeAttribute("src");
  video.load();
  videoName.textContent = "";
  playback.attach(null);
  syncVideoPane();
}

function syncVideoPane(): void {
  const wanted = timeline ? fileName(timeline.scene.metadata.source.path) : "";
  videoMissing.hidden = videoLoaded;
  videoMissingText.textContent = wanted
    ? `The clay still plays without it. This scene came from ${wanted}.`
    : "";
}

async function openFiles(files: Iterable<File>): Promise<void> {
  try {
    const { scene, video: source } = await readFiles(files);
    toast.hidden = true;
    const replacing = timeline !== null;
    if (scene) setScene(scene);
    if (source) setVideo(source);
    // A new scene on its own must not keep playing against the previous clip's footage.
    else if (scene && replacing) clearVideo();
    if (!scene && source && !timeline) {
      showToast(`${source.name} is ready. Now drop its scene file.`);
    }
  } catch (error) {
    showError(error);
  }
}

video.addEventListener("loadedmetadata", () => {
  if (!video.videoWidth) return;
  stage.style.setProperty("--video-aspect", String(video.videoWidth / video.videoHeight));
});

video.addEventListener("error", () => {
  if (!video.getAttribute("src")) return;
  const name = videoName.textContent || "that video";
  clearVideo();
  showToast(`This browser can't play ${name}. Convert it: ffmpeg -i clip -c:v libx264 -an clip.mp4`);
});

// Controls

playButton.addEventListener("click", () => playback.toggle());

scrub.addEventListener("pointerdown", () => {
  scrubbing = true;
});
scrub.addEventListener("input", () => {
  playback.seek((Number(scrub.value) / 1000) * playback.duration);
});
window.addEventListener("pointerup", () => {
  scrubbing = false;
});

views.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-view]");
  if (!button) return;
  world.setView(button.dataset.view as ViewName);
  syncViewButtons();
});

function setRate(rate: number): void {
  playback.rate = rate;
  for (const button of rates.querySelectorAll<HTMLButtonElement>("button")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.rate) === rate));
  }
}

rates.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-rate]");
  if (button) setRate(Number(button.dataset.rate));
});

function toggleDebug(): void {
  const enabled = !debug.enabled;
  debug.setEnabled(enabled);
  debugToggle.setAttribute("aria-pressed", String(enabled));
  lastSegment = -1;
}

debugToggle.addEventListener("click", toggleDebug);

document.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest("[data-action='pick']");
  if (target) picker.click();
});

picker.addEventListener("change", () => {
  if (picker.files?.length) void openFiles(picker.files);
  picker.value = "";
});

// Drag and drop

let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth++;
  veil.hidden = false;
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(dragDepth - 1, 0);
  if (dragDepth === 0) veil.hidden = true;
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  veil.hidden = true;
  if (event.dataTransfer?.files.length) void openFiles(event.dataTransfer.files);
});

// Splitter

function setSplit(percent: number): void {
  const clamped = Math.max(20, Math.min(80, percent));
  document.documentElement.style.setProperty("--split", `${clamped}%`);
  splitter.setAttribute("aria-valuenow", String(Math.round(clamped)));
}

splitter.addEventListener("pointerdown", (event) => {
  splitter.setPointerCapture(event.pointerId);
  splitter.dataset.dragging = "";
});
splitter.addEventListener("pointermove", (event) => {
  if (!splitter.hasPointerCapture(event.pointerId)) return;
  const box = stage.getBoundingClientRect();
  setSplit(((event.clientX - box.left) / box.width) * 100);
});
splitter.addEventListener("pointerup", () => {
  delete splitter.dataset.dragging;
});
splitter.addEventListener("keydown", (event) => {
  const step = event.key === "ArrowLeft" ? -2 : event.key === "ArrowRight" ? 2 : 0;
  if (!step) return;
  event.preventDefault();
  event.stopPropagation();
  setSplit(Number(splitter.getAttribute("aria-valuenow")) + step);
});

// Keyboard

window.addEventListener("keydown", (event) => {
  if (!timeline || event.metaKey || event.ctrlKey || event.altKey) return;
  const frameStep = 1 / (timeline.scene.metadata.source.fps || 30);
  const onButton = (event.target as HTMLElement).tagName === "BUTTON";
  switch (event.key) {
    case " ":
      if (onButton) return;
      event.preventDefault();
      playback.toggle();
      break;
    case "ArrowLeft":
      event.preventDefault();
      playback.seek(playback.time - SECOND_SKIP);
      break;
    case "ArrowRight":
      event.preventDefault();
      playback.seek(playback.time + SECOND_SKIP);
      break;
    case ",":
      playback.pause();
      playback.seek(playback.time - frameStep);
      break;
    case ".":
      playback.pause();
      playback.seek(playback.time + frameStep);
      break;
    case "1":
    case "2":
    case "3":
      world.setView((["chase", "aerial", "top"] as const)[Number(event.key) - 1]!);
      syncViewButtons();
      break;
    case "d":
    case "D":
      toggleDebug();
      break;
    default:
  }
});

// Frame loop

function frame(now: number): void {
  const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now;
  playback.tick(now);
  const time = playback.time;
  const readout = world.frame(time, dt);

  if (timeline && readout) {
    const speed = Math.round(readout.speed * KMH);
    hudSpeed.textContent = String(speed);
    hudMeta.textContent = `${readout.objects} ${readout.objects === 1 ? "object" : "objects"}`;
    timeNow.textContent = formatTime(time);
    if (!scrubbing) scrub.value = String(Math.round((time / timeline.duration) * 1000));
    scrub.setAttribute("aria-valuetext", formatTime(time));
    scrubber.draw(time);
    if (readout.segment !== lastSegment) {
      lastSegment = readout.segment;
      debug.describe(timeline, readout.segment);
    }
    debug.draw(world, readout.samples, timeline, readout.segment);
    playButton.toggleAttribute("data-playing", playback.playing);
    playButton.setAttribute("aria-label", playback.playing ? "Pause" : "Play");
  }
  requestAnimationFrame(frame);
}

// Start

async function start(): Promise<void> {
  await document.fonts.ready;
  requestAnimationFrame(frame);

  const params = new URLSearchParams(location.search);
  const sceneUrl = params.get("scene");
  const videoUrl = params.get("video");
  if (sceneUrl) {
    try {
      setScene(await fetchScene(sceneUrl));
      if (videoUrl) setVideo({ url: videoUrl, name: fileName(videoUrl) });
    } catch (error) {
      showError(error);
    }
    return;
  }

  const latest = await findLatest();
  if (latest) {
    openLatest.hidden = false;
    openLatest.textContent = `Open latest: ${latest.video.name}`;
    openLatest.addEventListener("click", () => {
      setScene(latest.scene);
      setVideo(latest.video);
    });
  }
}

if (import.meta.env.DEV) {
  // Lets a browser console or test drive exact frames while the tab is throttled.
  Object.assign(window, { clayview: { world, playback } });
}

void start();
