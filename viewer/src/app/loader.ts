import { parseScene, SceneError, type SceneFile } from "../data/scene";

export interface VideoSource {
  url: string;
  name: string;
}

export interface LoadedFiles {
  scene: SceneFile | null;
  video: VideoSource | null;
}

const VIDEO_EXTENSIONS = /\.(mp4|m4v|mov|webm|ogv|mkv)$/i;

/** Sort dropped or picked files into a scene and a video. */
export async function readFiles(files: Iterable<File>): Promise<LoadedFiles> {
  const result: LoadedFiles = { scene: null, video: null };
  let unknown = 0;
  for (const file of files) {
    if (file.name.endsWith(".json")) {
      result.scene = parseScene(await file.text());
    } else if (file.type.startsWith("video/") || VIDEO_EXTENSIONS.test(file.name)) {
      result.video = { url: URL.createObjectURL(file), name: file.name };
    } else {
      unknown++;
    }
  }
  if (unknown && !result.scene && !result.video) {
    throw new SceneError("Drop a .scene.json file and a video.");
  }
  return result;
}

export async function fetchScene(url: string): Promise<SceneFile> {
  const response = await fetch(url);
  if (!response.ok) throw new SceneError(`Couldn't load ${url} (${response.status}).`);
  return parseScene(await response.text());
}

export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** The pipeline's latest run, if this page is being served alongside it. */
export async function findLatest(): Promise<{ scene: SceneFile; video: VideoSource } | null> {
  try {
    const response = await fetch("out/latest.scene.json");
    const type = response.headers.get("content-type") ?? "";
    if (!response.ok || type.includes("text/html")) return null;
    const scene = parseScene(await response.text());
    const name = fileName(scene.metadata.source.path);
    const extension = name.slice(name.lastIndexOf("."));
    return { scene, video: { url: `out/latest${extension}`, name } };
  } catch {
    return null;
  }
}
