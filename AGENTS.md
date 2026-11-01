# clayview: agent guide

clayview turns car camera footage into a Tesla-style live clay 3D view of the traffic around you. It is a personal toy, not a driver aid: nothing here drives or should be trusted to.

## Status

Phase 1, recorded clips only: no hardware and no live camera. A Python pipeline turns a clip into a scene file, and a static Three.js viewer plays that file beside the video. The projection and tracking core is kept free of YOLO and OpenCV so it can run on a Pi later.

`docs/camera-layout/` will hold the camera placement study. Another agent owns that folder; leave it alone.

## Stack

- **Pipeline:** Python 3.12, `uv`, `ruff`, `pytest`. Ultralytics YOLO (`yolo11n` by default) with ByteTrack, OpenCV.
- **Viewer:** TypeScript (strict), Vite, Three.js, `bun`. No framework.
- **Tools:** versions in `.mise.toml`. ffmpeg for clip conversion.

The repo is private because Ultralytics is AGPL-3.0. Don't make it public or ship it without dealing with that.

## Commands

```sh
uv sync                                  # install the pipeline
uv run clayview process clips/clip.mp4   # clip to out/clip.scene.json
uv run clayview stats out/clip.scene.json
uv run pytest
uv run ruff check && uv run ruff format --check

cd viewer
bun install
bun run dev        # http://localhost:5199, serves out/ for "Open latest"
bun run test
bun run typecheck
bun run build      # typecheck, then dist/
```

Detections are cached in `out/<clip>.raw.npz`, so changing a camera flag takes seconds. `--fresh` redoes them; changing detection code needs it.

## Layout

```
src/clayview/
  core/          numpy only. A test enforces this.
    camera.py      pinhole road-plane model
    projection.py  box to road plane; ground and size-prior fusion
    smoother.py    ego-compensated Kalman filter, RTS pass
    ego.py         ego speed and yaw from road flow
    horizon.py     per-shot calibration: FoE, car widths, hood row
    cuts.py        hard-cut detection
    tracks.py      stitching, filtering, heading, confidence, flags
    scene.py       builds the scene file
    stats.py       `clayview stats`
  detect.py      the only module that imports Ultralytics
  video.py       OpenCV: frames, cut thumbnails, background flow
  pipeline.py    detection pass and its cache
  cli.py
viewer/src/
  data/scene.ts  scene contract; sampling is a pure function of time
  world/         stage, road, procedural models, traffic, camera views
  app/           playback clock, loading, scrubber, debug overlay
  main.ts        wiring
clips/ out/ models/   gitignored
```

## How it fits together

The scene file is the only link between the halves. Its shape is defined by `core/scene.py` on one side and `viewer/src/data/scene.ts` on the other: change both, and bump `SCENE_VERSION` in both for a breaking change.

Ego frame: `x` right, `y` forward, metres, origin on the road under the camera. Heading is radians, 0 is ego's direction, positive turns right. `x`, `y` are the object's centre, not the near face.

## Conventions

- **Check against footage.** Synthetic data passed while real clips produced seven false cuts, reversed parked cars and an ego speed a third too low. When output looks wrong, measure it on a real clip; add the failure as a test.
- Core stays numpy-only and takes plain dataclasses. Anything needing a model or a frame goes outside `core/`.
- Viewer state at time t is a pure function of t. No stateful fades or per-frame integration, so scrubbing can't leave ghosts.
- Clay palette lives in `world/palette.ts`. Terracotta is reserved for ego.
- No external 3D assets, and nothing loaded from a CDN: the viewer has to work from plain static files.
- Strict TS, explicit return types on exports, named exports. Type hints on Python signatures, `pathlib`, dataclasses for records.
- Conventional commits, one logical change each. Never commit a video, weights or scene file.

## Out of scope

Self-driving, driver assistance or anything safety-related. Live cameras and hardware, until this phase is done. Fetching from YouTube. Lane detection: the road in the viewer is decoration, and must stay described as such.
