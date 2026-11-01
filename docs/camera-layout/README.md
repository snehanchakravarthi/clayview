# Camera layout study

Where cameras, the computer and the wires should go once clayview moves off a recorded clip.

- Live planner: https://claude.ai/artifact/Xv1vJQCUXpRduvDdLMFe1c (private)
- Same page in this repo: `planner.html`. Open it in a browser. It only needs internet for fonts.
- Source: `src/model.js` (geometry and error model), `src/template.html` (the page), `src/build.js` (writes `planner.html`), `src/score.js` (prints the table below).

```sh
node docs/camera-layout/src/build.js
node docs/camera-layout/src/score.js
```

This is a geometry model. No rig here has been tested on a car.

## Results

Tall SUV, 640 px detector input, cars, 0.25° tilt error after IMU correction. "Placed" means the position error is under 15% of the distance, or 0.5 m when close.

| Layout | Cameras | Door zone 0–3 m | Near 3–15 m | Mid 15–40 m | Far 40–80 m | Scene objects placed | Max fps per camera |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phone on the dash | 1 | 0% | 2% | 14% | 0% | 3/26 | n/a |
| Four surround, 120° | 4 | 53% | 95% | 44% | 0% | 12/26 | 34 |
| Four corners, 150° | 4 | 59% | 100% | 28% | 0% | 16/26 | 34 |
| **Four, fisheye mirrors** | 4 | 72% | 98% | 40% | 0% | 17/26 | 34 |
| Five, add a long lens | 5 | 77% | 98% | 42% | 2% | 17/26 | 27 |
| Six inside the glass | 6 | 30% | 97% | 78% | 2% | 15/26 | 23 |
| Six on a roof rack | 6 | 21% | 98% | 99% | 14% | 15/26 | 23 |
| **Roof four plus fisheye mirrors** | 6 | 72% | 100% | 88% | 5% | 19/26 | 23 |
| Eight, Tesla style | 8 | 62% | 96% | 59% | 1% | 15/26 | 17 |

## What it says

1. **Height matters more than camera count.** Ground-contact distance error shrinks as the camera rises, and height also sees over nearby cars.
2. **The door zone needs low, wide lenses.** A roof camera can't see the road beside the car, and a windscreen camera can't see the road for the first several metres.
3. **Past 40 m, cars are seen but not placed.** Fade them in the clay view.
4. **A long front lens adds range but little accuracy.** Tilt error dominates at distance, and more pixels don't fix it.
5. **Skipping the IMU hurts most in the middle.** At 1° tilt error, mid-range placement falls to 7–31% on the multi-camera rigs.

## Recommendation

- **First build: four cameras.** 90° front, 100° rear, 160° on each mirror, about $410 in parts by rough estimate.
- **Most realistic: six cameras.** Roof front, rear, left and right, plus the two 160° mirror lenses, about $530.
- **Computer:** Raspberry Pi 5 with the 26 TOPS AI HAT+, under the front passenger seat.
- **Links:** USB 2 passive runs stay under the 5 m limit in the seat position. Runs through the door seal are the hard part.
- **Power:** fused switched 12 V, a 5 V 5 A converter, and a clean shutdown when the ignition goes off.

## Limits

- Tracking over time isn't modelled, which should help every rig.
- No learned depth is modelled, so the Tesla row scores this pipeline's method on that layout. It says nothing about Tesla's system.
- Fisheye edges are treated as usable. YOLO is weaker there, and splitting a fisheye into flat views costs compute.
- Rain, night and dirty lenses are ignored.
- The 16 px detection limit, 2 px box noise, 0.25° tilt error, cable routes and parts costs are assumptions, not measurements.
