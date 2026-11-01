# clayview

A Tesla-style clay 3D view of the traffic around a car, built from camera footage. Give it a dashcam clip and it plays the clip beside a matte clay model of the same scene: every car, truck, bus, motorcycle, bicycle and person, placed on the road around you.

It's a toy. It doesn't drive anything and shouldn't be trusted for anything.

Today it works on recorded clips only. The next phases, live cameras on a Pi, aren't built. The camera placement study and its interactive planner are in [`docs/camera-layout/`](docs/camera-layout/README.md).

## Setup

You need [uv](https://docs.astral.sh/uv/), [bun](https://bun.sh) and ffmpeg. With [mise](https://mise.jdx.dev), `mise trust && mise install` gets the pinned versions.

```sh
uv sync
cd viewer && bun install
```

The first run downloads the YOLO weights, the car view model and the street model into `models/`, about 1 GB in total.

## Run it on a clip

Put a clip in `clips/`, which is gitignored, and process it:

```sh
uv run clayview process clips/drive.mp4
```

This writes `out/drive.scene.json` and prints a summary of what it found. Then start the viewer:

```sh
cd viewer
bun run dev
```

Open <http://localhost:5199> and press **Open latest**. You can also drop a scene file and its video anywhere on the page, or use **Choose files**.

The clip has to be something your browser can play. H.264 MP4 is the safe choice. Old `.ogv` files need converting first:

```sh
ffmpeg -i clip.ogv -c:v libx264 -crf 23 -an clip.mp4
```

### Viewer controls

| Key | Action |
| --- | --- |
| Space | Play or pause |
| ← → | Skip 5 seconds |
| , . | Step one frame |
| 1 2 3 | Chase, aerial and top views |
| D | Debug: distances, flags, boxes on the video, horizon and calibration |

Drag the clay scene to orbit it. The bars on the timeline show how much traffic is in view, and terracotta ticks mark hard cuts.

The built viewer (`bun run build`) is plain static files with no backend. `?scene=<url>&video=<url>` opens a scene from any URL.

## How camera parameters work

The pipeline assumes a pinhole camera, level with the horizon row. A point on the road at image row `y` is this far ahead:

```
d = f * h / (y - horizon)      f: focal length in pixels, h: camera height
```

That is precise up close, but one pixel is worth `d²/(f*h)` metres, so it falls apart near the horizon. It also needs the bottom of the car to be visible. So each box gets a second estimate from a typical real size, `d = f * W / width_px`, and the two are blended by how much each can be trusted at that range. Objects that rely on the size estimate are flagged `size_prior`, and those with no usable ground contact are flagged `no_ground`.

Clips from unknown cameras are calibrated per shot, so a compilation can change camera at every cut.

| Flag | Default | What it does |
| --- | --- | --- |
| `--vfov` | `60` | Vertical field of view in degrees. **This sets the scale of every distance and can't be estimated from the clip.** A wider real lens than you've set makes everything read too far away. |
| `--height` | `auto` | Camera height in metres. `auto` works it out from the widths of rear-on cars. Falls back to 1.3. |
| `--horizon` | `auto` | Horizon row, in pixels or as a fraction under 1. `auto` uses the focus of expansion of the road flow, then car widths, then mid-frame. |
| `--pitch` | | Camera pitch in degrees, down is positive. Sets the horizon instead. |
| `--ego-speed` | `auto` | Speed in km/h, `auto` from road flow, or `off` to work in relative motion only. |
| `--max-range` | `120` | Clamp distance in metres. Objects past it are flagged `clamped`. |
| `--model` | `models/yolo11n.pt` | YOLO weights. |
| `--no-views` | | Skip the car view model. Faster, but headings get worse. |
| `--no-world` | | Skip the street model. No real road, trees or buildings. |
| `--imgsz` | `960` | Inference size. Higher finds smaller, further cars. |
| `--stride` | `1` | Process every Nth frame. |
| `--limit` | | Stop after this many seconds. |

Detections are cached, so re-running with a different camera flag takes seconds. Pass `--fresh` to redo them.

If the output looks wrong, press D in the viewer. The yellow line should sit on the real horizon, and the pink one on the top of the hood. `uv run clayview stats out/drive.scene.json` prints the numbers.

## The scene file

```json
{
  "version": 1,
  "metadata": { "source": {}, "camera": {}, "model": {}, "egoMotion": {} },
  "segments": [{ "index": 0, "start": 0, "end": 120, "camera": {} }],
  "frames": [
    {
      "frame": 1875, "t": 62.5, "segment": 0,
      "ego": { "speed": 12.05, "yaw": -0.00431 },
      "objects": [
        {
          "id": 48, "class": "car", "x": -8.35, "y": 9.82,
          "heading": -0.349, "confidence": 0.592, "flags": [],
          "box": [0.0983, 0.6211, 0.1868, 0.7013]
        },
        {
          "id": 50, "class": "car", "x": -7.24, "y": 0.33,
          "heading": 0.248, "confidence": 0.194, "flags": ["held", "low_conf"]
        }
      ]
    }
  ]
}
```

Positions are in metres in the ego frame: `x` right, `y` forward, measured to the object's centre. Heading is in radians over the ground: 0 is ego's direction, positive turns right. `box` is the detector's box as fractions of the frame, and is missing on `held` frames.

Flags are `size_prior`, `no_ground`, `truncated` (cut by a frame edge), `clamped`, `low_conf` and `held` (no detection this frame; the position comes from the filter).

## Known limits

- **Distances are estimates with an unknown scale.** `--vfov` is a guess for any clip you didn't film yourself. Expect the relative layout to be good and absolute metres to be off by 20% or more.
- **Lenses are assumed perfect.** Wide dashcam lenses bend straight lines, which matters most at the edges of the frame.
- **The road is assumed flat.** Hills move the true horizon, and calibration is per shot, not per frame.
- **Ego speed comes from road texture.** It has no ground truth behind it. Bare asphalt, night and rain will all hurt it.
- **Headings are approximate.** They combine direction of travel, box shape, the road direction from ego's own path and an image model that reads the angle a car is seen from. On 32 hand-labelled cars from the test clips, the median error is about 32° and 12% of cars face the wrong way. That's a small sample on one dashcam. The view model is often wrong on cars partly hidden by a dash mount, and small, far cars are weak.
- **The street is rebuilt from one forward camera.** Curbs, paint, trees, poles and buildings are projected onto a flat road and joined along ego's estimated path, so errors in speed, yaw or calibration bend the map. Paint near the car can be about a metre off and arrows can smear. Anything the camera never saw clearly is missing, and tree heights run low when a canopy leaves the frame.
- **Only one camera sees forward.** Things that pass you coast briefly and fade out. Nothing is known behind or beside you.
- **`yolo11n` is small.** It misses distant objects and sometimes confuses cars and trucks. `--model` takes a larger one.

## Licence

This repo is private because [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) is AGPL-3.0. Sort that out before making it public or shipping it. Clips and model weights are never committed.

The street model is Mask2Former trained on [Mapillary Vistas](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic). Its Hugging Face licence is listed as `other`, and Vistas is a non-commercial research dataset, so check the terms before using this for anything but a personal project.

The car view model is the ConvNeXt from [Car Full View Dataset](https://www.mdpi.com/2079-9292/12/24/4947) (Catruna et al., 2023), Apache-2.0 on [Hugging Face](https://huggingface.co/fort-cyber/Car-orientation-image). `clayview` checks its SHA-256 after download.
