# Focal length from footage: what was tried

Goal: estimate vertical field of view for clips without a measured `vfov`, so they stop running at
the 60° default. The hold-out test is comma2k19 (51.3°, 910 px) and md_i270 (9.2°, 4480 px), run
with their `vfov` hidden. Nothing here passed it, so the pipeline still uses the default.

`probe.py` reproduces every number below from the pulled eval caches:

```sh
uv run python tools/eval/eval.py pull
uv run python tools/focal/probe.py comma2k19 md_i270 bled_rollei   # add --skip-rotation for speed
```

## Why most signals can't see focal length

To first order, scaling f is an exact symmetry of what a forward camera sees. Depths along the road
scale with f, angles scale with 1/f, and lateral offsets don't change. So these all measure camera
height and say nothing about f:

- car widths, `width / (row - horizon) = W / h`;
- lane widths, the same;
- the ego's lateral motion: a lane change's yaw is `lateral accel / speed`, and f cancels;
- lateral acceleration in turns, `v ω`: speed scales with f, yaw rate with 1/f.

Only three things break the symmetry: a known length along the road (dash cycle, car length), a
known absolute angle (a car's appearance, a circle), or the wide-angle second-order terms of the
pinhole. Each was tried.

## Ruled out

**Car view model as an angle sensor.** Traffic runs parallel to ego, so a car at column x is seen
from `atan((x - FoE x) / f)`. The view model's eight 45° classes are confident and sticky: cars
under 20° off read as dead rear, near side-lane cars read 45°. The likelihood over vfov is monotonic
on every clip, peaking at the 4° or 100° end of the grid (comma2k19: 100°, md_i270: 4°).

**Rotation field self-calibration.** The full pinhole rotation flow has `x²/f` and `xy/f` terms.
Fitted per frame with the FoE fixed, comma2k19 scores best near 40–45° and md_i270 at ≤ 4°,
with the narrow side of each curve almost flat. Adding per-frame rolling-shutter shear columns
(`u ∝ y`, `v ∝ y`) makes the score exactly independent of f on every clip. The quadratic rotation
terms are radial about the principal point. Forward translation is radial about the FoE, with
free per-point depth. So the f signal is only the FoE's offset from the image centre, i.e. camera
pitch, and it dies once shear is modelled. This is degenerate for forward-driving footage.

**Car length from box corners.** For a car wholly to one side of the FoE, the outer box edge is the
near rear corner at depth Z and the inner edge is the far front corner at Z + L, so
`(x_outer - w_rear) / x_inner = 1 + L (row - horizon) / (f h)`. Per box this came out at 13.5° on
comma2k19 and 6.5° on md_i270, with most samples at s ≈ 0. Two reasons. The rear-width scale
`w_rear` comes from the same "rear-on" boxes, which are already slightly oblique, so it absorbs
the side. And md_i270's near boxes are narrower than rear-on (0.57–0.9 of `w_rear`). Box aspect
against column has the same problem in another form. comma2k19 fits a rounded footprint with
L/W ≈ 2.9, while md_i270 rises like a rectangle at 3° off-axis. No one shape prior fits both.

**Pitch-coupled roll.** A camera pitched by θ sees vehicle yaw partly as roll:
`roll rate / yaw shift = dy / f²`, with dy = image centre row − horizon. Shear columns keep rolling
shutter out of the roll term. On comma2k19 this gives 956 px, 49.1° (truth 51.3°), but at ±50% in
slope over 92 turning frames. md_i270's horizon is 4 px from centre, so the signal is absent. Over
all clips, 9 of 22 slopes have the wrong sign for their horizon, so body roll, mount yaw coupling
or the principal point dominate. sf_bay_bridge comes out at 119°.

| clip | roll estimate | plausible |
|---|---|---|
| comma2k19 | 49.1° | yes (truth 51.3°) |
| kose_winter | 53.1° | yes |
| ljubljana_fog_night | 72.4° | doubtful |
| kranj_glare | wrong sign | — |
| doha_rain_360p | wrong sign | — |
| matanzas_portrait | 48.5° | low for portrait |
| christchurch_left | wrong sign | — |
| mirpur_dhaka | wrong sign | — |
| urdu_dhaka | 51.2° | yes |
| leh_bike | wrong sign | — |
| bled_rollei | 59.1° | doesn't explain the 0.64 speed ratio (needs ~42°) |
| wintry_wipers | 36.3° | narrow but possible |
| montblanc_tunnel | 33.3° | narrow but possible |
| touareg_interior | wrong sign | — |
| dashcam_pip | 58.0° | yes |
| md_i270 | no signal (dy 4 px) | — (truth 9.2°) |
| krakow_city | wrong sign | — |
| bangalore_iisc | wrong sign | — |
| hahndorf_town | wrong sign | — |
| plitvice_highway | 66.5° | timelapse, ignored |
| sf_bay_bridge | 118.6° | no |
| park_ridge_suburb | 76.2° | doubtful |

## What does work, and why it isn't general

**Dash period.** md_i270's 9.2° comes from this: dashes repeat every 0.0021 in 1/(row − horizon),
which is `cycle / (f h)`, with h from car widths. But the cycle is regional: 12.19 m on US
interstates, 14.6 m on California's 12 + 36 ft standard (comma2k19 is on a California highway),
18 m on German motorways, and other values elsewhere. The street labels are lane paint on a 32 px
grid, too coarse to resolve dashes far away, so this needs frames. It can't run as a default
without knowing the country. It would make sense as an opt-in flag such as `--dash-cycle 12.19`.

## Next steps that might work

- Dash period with a user-supplied or country-supplied cycle, read from frames in `world.py`.
- A continuous car-orientation regressor instead of the 8-class model; the angle-sensor idea is
  sound, the classifier's bins are the problem.
- Wheels: a parallel car's wheel is a circle seen at the bearing angle, so its ellipse aspect is
  `sin(bearing)`. That's an absolute angle from a circle, with no shape prior.
- bled_rollei: the overlay asks for ~42° at the image centre, while the clip's rotation fit prefers
  ~70°. That fits a barrel-distorted action camera whose centre is magnified, and suggests fixing it
  through the lens model (#8's distortion gate rejects bled) rather than through focal length.
