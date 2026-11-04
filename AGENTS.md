# clayview: agent guide

clayview turns car camera footage into a Tesla-style live clay 3D view of the traffic around you. It is a personal toy, not a driver aid: nothing here drives or should be trusted to.

## Status

Phase 1, recorded clips only: no hardware and no live camera. A Python pipeline turns a clip into a scene file, and a static Three.js viewer plays that file beside the video. The projection and tracking core is kept free of YOLO and OpenCV so it can run on a Pi later.

`docs/camera-layout/` holds the camera placement study and its interactive planner. Edit `src/` there and rebuild `planner.html` with `node docs/camera-layout/src/build.js`.

## Stack

- **Pipeline:** Python 3.12, `uv`, `ruff`, `pytest`. Ultralytics YOLO (`yolo11n` by default) with ByteTrack, OpenCV.
- **Viewer:** TypeScript (strict), Vite, Three.js, `bun`. No framework.
- **Tools:** versions in `.mise.toml`. ffmpeg for clip conversion.

The repo is public. Ultralytics is AGPL-3.0 and the street model's training data (Mapillary Vistas) is non-commercial, so don't ship a hosted service or a binary without dealing with that. No licence has been chosen for clayview's own code yet; leave that to the owner.

## Commands

```sh
uv sync                                  # install the pipeline
uv run clayview process clips/clip.mp4   # clip to out/clip.scene.json
uv run clayview stats out/clip.scene.json
uv run pytest
uv run ruff check && uv run ruff format --check
uv run python tools/eval/eval.py fetch    # clips from many cameras, into clips/eval/
uv run python tools/eval/eval.py run      # process and score them; `score` rescores
uv run python tools/eval/eval.py pull     # clips and caches from the eval-data release; use this in the cloud
uv run python tools/eval/eval.py pack     # on a machine that has run everything: tars for that release

cd viewer
bun install
bun run dev        # http://localhost:5199, serves out/ for "Open latest"
bun run test
bun run typecheck
bun run build      # typecheck, then dist/
```

Detections, car views and street labels are cached in `out/<clip>.raw.npz`, `.views.npz` and `.seg.npz`, so changing a camera flag takes seconds. `--fresh` redoes them; changing detection or model code needs it.

Caches are keyed on the video's bytes, so they travel with the clip. Cloud sessions should `pull` rather than `fetch`: Wikimedia rate-limits cloud addresses, and without a GPU the models take about 27 minutes a clip. With pulled caches, the whole eval rescores in minutes. A change to detection or the models invalidates the caches, so run that one where there's a GPU.

## Layout

```
src/clayview/
  core/          numpy only. A test enforces this.
    camera.py      pinhole road-plane model
    projection.py  box to road plane; ground and size-prior fusion
    smoother.py    ego-compensated Kalman filter, RTS pass
    ego.py         ego speed and yaw from road flow
    mount.py       the car's yaw apart from the camera turning on its mount
    horizon.py     per-shot calibration: FoE, car widths, hood row
    cuts.py        hard-cut detection
    heading.py     heading from box shape, motion, ego's path and car views
    tracks.py      stitching, filtering, confidence, flags
    scene.py       builds the scene file
    stats.py       `clayview stats`
  detect.py      the only module that imports Ultralytics
  orient.py      the car view model, the only module that imports timm
  segment.py     the street model, the only module that imports transformers
  world.py       the static street: road edges, paint, trees, poles, buildings (needs OpenCV)
  video.py       OpenCV: frames, cut thumbnails, background flow
  pipeline.py    detection pass and its cache
  cli.py
tools/eval/      runs the pipeline over clips.json and scores it against known truth
viewer/src/
  data/scene.ts  scene contract; sampling is a pure function of time
  world/         stage, the clay street (surface, flora, props, massing), traffic, camera views
  app/           playback clock, loading, scrubber, debug overlay
  main.ts        wiring
tools/eval/       clips.json lists clips from other cameras and what's known about each
clips/ out/ models/   gitignored
```

## How it fits together

The scene file is the only link between the halves. Its shape is defined by `core/scene.py` on one side and `viewer/src/data/scene.ts` on the other: change both, and bump `SCENE_VERSION` in both for a breaking change.

Ego frame: `x` right, `y` forward, metres, origin on the road under the camera. Heading is radians, 0 is ego's direction, positive turns right. `x`, `y` are the object's centre, not the near face.

## Conventions

- **Check against footage.** Synthetic data passed while real clips produced seven false cuts, reversed parked cars and an ego speed a third too low. When output looks wrong, measure it on a real clip; add the failure as a test.
- **Check against other cameras.** Run `tools/eval/eval.py` before calling a pipeline change done. comma2k19 has speed, pose and radar truth; tuning on one dashcam is how wipers became cuts and a hood reflection cut ego speed by 40%.
- Anything counted in frames should be a time: clips run from 20 to 60 fps, and `--stride` changes the rate again.
- Core stays numpy-only and takes plain dataclasses. Anything needing a model or a frame goes outside `core/`.
- Viewer state at time t is a pure function of t. No stateful fades or per-frame integration, so scrubbing can't leave ghosts.
- Clay palette lives in `world/palette.ts`. Terracotta is reserved for ego.
- No external 3D assets, and nothing loaded from a CDN: the viewer has to work from plain static files.
- Strict TS, explicit return types on exports, named exports. Type hints on Python signatures, `pathlib`, dataclasses for records.
- Conventional commits, one logical change each. Never commit a video, weights or scene file.

## Out of scope

Self-driving, driver assistance or anything safety-related. Live cameras and hardware, until this phase is done. Fetching from YouTube. Pretending the rebuilt street is survey accurate: it comes from one forward camera and ego's estimated path, and must stay described that way.
