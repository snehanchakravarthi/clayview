// Geometry and error model for the camera planner. No DOM in here.
// Ego frame: x forward, y left, metres. Yaw 0 is forward, positive turns left.
const DEG = Math.PI / 180;

const VEHICLES = {
  hatch: { label: 'Hatchback', L: 3.9, W: 1.75, H: 1.55 },
  sedan: { label: 'Sedan', L: 4.5, W: 1.78, H: 1.48 },
  suv: { label: 'Tall SUV (Thar Roxx size)', L: 4.43, W: 1.87, H: 1.92 },
};

const CLASSES = {
  car: { label: 'Car', w: 1.8, l: 4.4, h: 1.5 },
  auto: { label: 'Auto', w: 1.4, l: 2.7, h: 1.7 },
  bike: { label: 'Two-wheeler', w: 0.75, l: 1.9, h: 1.3 },
  person: { label: 'Person', w: 0.5, l: 0.4, h: 1.7 },
  truck: { label: 'Truck', w: 2.5, l: 8, h: 3.2 },
  bus: { label: 'Bus', w: 2.6, l: 11, h: 3.2 },
};

const DEFAULT_OPTS = {
  inputPx: 640, // detector input width
  minPx: 16, // smallest object width the detector finds reliably
  pxErr: 2, // box edge noise in detector pixels
  pitchErrDeg: 0.25, // road slope and body pitch left after IMU correction
  sizeErr: 0.2, // real width varies this much within a class
  chipFps: 135, // Hailo-8 vendor 270 fps at 640, halved for real pipelines
};

const SCENARIOS = {
  highway: {
    label: 'Highway',
    note: 'Fast traffic at range. Long lens territory.',
    objects: [
      { c: 'car', x: 35, y: 0, name: 'Car ahead' },
      { c: 'car', x: 18, y: 3.5, name: 'Car in left lane' },
      { c: 'truck', x: 58, y: -3.5, name: 'Truck far ahead' },
      { c: 'car', x: -14, y: 3.5, name: 'Overtaker, rear left' },
      { c: 'car', x: -30, y: 0, name: 'Car behind' },
    ],
  },
  jam: {
    label: 'City jam',
    note: 'Everything inside 8 m, two-wheelers at the doors.',
    objects: [
      { c: 'auto', x: 5, y: 0, name: 'Auto ahead' },
      { c: 'bus', x: 1, y: -3.3, name: 'Bus on the right' },
      { c: 'bike', x: 0.5, y: 1.7, name: 'Bike at left door' },
      { c: 'bike', x: -3, y: 1.6, name: 'Bike, rear left' },
      { c: 'car', x: -6.5, y: 0, name: 'Car behind' },
      { c: 'person', x: 8, y: 3, name: 'Pedestrian, front left' },
    ],
  },
  behind: {
    label: 'Hidden car',
    note: 'A truck ahead hides what is beyond it.',
    objects: [
      { c: 'truck', x: 14, y: 0, name: 'Truck ahead' },
      { c: 'car', x: 26, y: 0, name: 'Car behind the truck' },
      { c: 'bike', x: 16, y: -2.4, name: 'Bike beside the truck' },
      { c: 'car', x: 20, y: 3.5, name: 'Car in left lane' },
      { c: 'car', x: -12, y: 0, name: 'Car behind us' },
    ],
  },
  junction: {
    label: 'Junction',
    note: 'Cross traffic from both sides.',
    objects: [
      { c: 'car', x: 18, y: 16, h: -90, name: 'Cross traffic, left' },
      { c: 'car', x: 12, y: -20, h: 90, name: 'Cross traffic, right' },
      { c: 'truck', x: 24, y: -34, h: 90, name: 'Truck on the crossing road' },
      { c: 'person', x: 7, y: -3, name: 'Pedestrian crossing' },
      { c: 'bike', x: 4, y: 3, h: -30, name: 'Bike cutting across' },
    ],
  },
  parking: {
    label: 'Parking',
    note: 'Under 3 m. Where tyre contact disappears.',
    objects: [
      { c: 'car', x: 0, y: 2.6, name: 'Parked car, left' },
      { c: 'car', x: 0, y: -2.6, name: 'Parked car, right' },
      { c: 'person', x: -3.6, y: 0.5, name: 'Person behind us' },
      { c: 'bike', x: -2.4, y: -1.5, name: 'Bike behind right' },
      { c: 'car', x: 5.5, y: 0, name: 'Car ahead' },
    ],
  },
};

// u,v are fractions of half-length and half-width. zf is a fraction of height.
const FRONT = (fov, name = 'Front') => ({ name, u: 0.28, v: 0, zf: 0.78, yaw: 0, fov, pitch: 0, mount: 'cabin' });
const REAR = (fov) => ({ name: 'Rear', u: -0.86, v: 0, zf: 0.72, yaw: 180, fov, pitch: 3, mount: 'cabin' });
const MIRROR = (side, fov) => ({
  name: side > 0 ? 'Left mirror' : 'Right mirror',
  u: 0.34, v: 1.1 * side, zf: 0.58, yaw: 90 * side, fov, pitch: 18, mount: 'exterior',
});

const ROOFCAM = (name, u, v, yaw, fov, pitch) => ({ name, u, v, zf: 1.03, yaw, fov, pitch, mount: 'roof' });
const CABIN_SIDE = (name, u, side, yaw) => ({ name, u, v: 0.78 * side, zf: 0.72, yaw: yaw * side, fov: 90, pitch: 6, mount: 'cabin' });

const RIGS = [
  {
    id: 'phone', name: 'Phone on the dash', short: '1 phone',
    blurb: 'The recorded-clip prototype. One view ahead and nothing around you.',
    cams: [{ name: 'Dash phone', u: 0.22, v: 0, zf: 0.68, yaw: 0, fov: 70, pitch: 0, mount: 'cabin' }],
  },
  {
    id: 'surround4', name: 'Four surround, 120°', short: '4 at 120°',
    blurb: 'One 120° lens per side. The obvious layout, with gaps at the corners.',
    cams: [FRONT(120), REAR(120), MIRROR(1, 120), MIRROR(-1, 120)],
  },
  {
    id: 'corner4', name: 'Four corners', short: '4 corners',
    blurb: 'Your first idea: a 150° lens low on each corner of the car.',
    cams: [
      { name: 'Front left', u: 0.97, v: 0.92, zf: 0.4, yaw: 45, fov: 150, pitch: 10, mount: 'exterior' },
      { name: 'Front right', u: 0.97, v: -0.92, zf: 0.4, yaw: -45, fov: 150, pitch: 10, mount: 'exterior' },
      { name: 'Rear left', u: -0.97, v: 0.92, zf: 0.4, yaw: 135, fov: 150, pitch: 10, mount: 'exterior' },
      { name: 'Rear right', u: -0.97, v: -0.92, zf: 0.4, yaw: -135, fov: 150, pitch: 10, mount: 'exterior' },
    ],
  },
  {
    id: 'value4', name: 'Four, fisheye mirrors', short: '4 fisheye', tag: 'First build',
    blurb: 'Narrower front and rear, 160° on each mirror looking down at the doors.',
    cams: [FRONT(90), REAR(100), MIRROR(1, 160), MIRROR(-1, 160)],
  },
  {
    id: 'value5', name: 'Five, add a long lens', short: '5 long',
    blurb: 'The same idea with a 45° lens for range ahead.',
    cams: [FRONT(120, 'Front wide'), { ...FRONT(45, 'Front long'), v: 0.08 }, REAR(120), MIRROR(1, 160), MIRROR(-1, 160)],
  },
  {
    id: 'glass6', name: 'Six inside the glass', short: '6 in cabin',
    blurb: 'No exterior work at all. Every camera sticks to the inside of a window.',
    cams: [
      { name: 'Front', u: 0.28, v: 0, zf: 0.78, yaw: 0, fov: 70, pitch: 0, mount: 'cabin' },
      CABIN_SIDE('Front left', 0.22, 1, 60), CABIN_SIDE('Front right', 0.22, -1, 60),
      CABIN_SIDE('Rear left', -0.72, 1, 125), CABIN_SIDE('Rear right', -0.72, -1, 125),
      { name: 'Rear', u: -0.86, v: 0, zf: 0.72, yaw: 180, fov: 110, pitch: 3, mount: 'cabin' },
    ],
  },
  {
    id: 'roof6', name: 'Six on a roof rack', short: '6 on roof',
    blurb: 'All on the roof edge, one cable into the car. Sees over traffic but not down at the doors.',
    cams: [
      ROOFCAM('Front', 0.33, 0, 0, 70, 2),
      ROOFCAM('Front left', 0.3, 0.99, 55, 90, 14), ROOFCAM('Front right', 0.3, -0.99, -55, 90, 14),
      ROOFCAM('Rear left', -0.93, 0.99, 125, 90, 14), ROOFCAM('Rear right', -0.93, -0.99, -125, 90, 14),
      ROOFCAM('Rear', -0.96, 0, 180, 110, 6),
    ],
  },
  {
    id: 'roofmirror6', name: 'Roof four plus fisheye mirrors', short: '4 roof + 2', tag: 'Most realistic',
    blurb: 'Height for the mid range, low fisheye lenses for the doors.',
    cams: [
      ROOFCAM('Roof front', 0.33, 0, 0, 70, 2), ROOFCAM('Roof rear', -0.96, 0, 180, 110, 6),
      ROOFCAM('Roof left', -0.3, 0.99, 90, 110, 14), ROOFCAM('Roof right', -0.3, -0.99, -90, 110, 14),
      MIRROR(1, 160), MIRROR(-1, 160),
    ],
  },
  {
    id: 'tesla8', name: 'Eight, Tesla style', short: '8 Tesla',
    blurb: 'An approximation of the layout: three forward, four side, one rear. Scored on this pipeline, not on Tesla’s.',
    cams: [
      { name: 'Front main', u: 0.28, v: 0, zf: 0.78, yaw: 0, fov: 50, pitch: 0, mount: 'cabin' },
      { name: 'Front wide', u: 0.28, v: 0.06, zf: 0.78, yaw: 0, fov: 120, pitch: 0, mount: 'cabin' },
      { name: 'Front long', u: 0.28, v: -0.06, zf: 0.78, yaw: 0, fov: 35, pitch: 0, mount: 'cabin' },
      { name: 'Left pillar', u: -0.02, v: 1.05, zf: 0.62, yaw: 65, fov: 90, pitch: 5, mount: 'exterior' },
      { name: 'Right pillar', u: -0.02, v: -1.05, zf: 0.62, yaw: -65, fov: 90, pitch: 5, mount: 'exterior' },
      { name: 'Left fender', u: 0.5, v: 1.05, zf: 0.45, yaw: 140, fov: 75, pitch: 3, mount: 'exterior' },
      { name: 'Right fender', u: 0.5, v: -1.05, zf: 0.45, yaw: -140, fov: 75, pitch: 3, mount: 'exterior' },
      { name: 'Rear', u: -1.02, v: 0, zf: 0.45, yaw: 180, fov: 130, pitch: 8, mount: 'exterior' },
    ],
  },
];

const ROOF = { front: 0.32, rear: -0.95 };

function placeCam(cam, veh) {
  return {
    ...cam,
    x: (cam.u * veh.L) / 2,
    y: (cam.v * veh.W) / 2,
    z: cam.zf * veh.H,
    yawR: cam.yaw * DEG,
    fovR: cam.fov * DEG,
    vfovR: cam.fov * DEG * (9 / 16),
    pitchR: cam.pitch * DEG,
  };
}

const angleDiff = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
const focalPx = (fovDeg, inputPx) => inputPx / (fovDeg * DEG); // pixels per radian, equidistant approximation

function detectRange(fovDeg, widthM, opts) {
  return (focalPx(fovDeg, opts.inputPx) * widthM) / opts.minPx;
}

// Slab test of the segment a->b against a centred box of half sizes hx, hy.
function segBox(ax, ay, bx, by, hx, hy) {
  let t0 = 0;
  let t1 = 1;
  const dx = bx - ax;
  const dy = by - ay;
  for (const [p, d, h] of [[ax, dx, hx], [ay, dy, hy]]) {
    if (Math.abs(d) < 1e-9) {
      if (Math.abs(p) > h) return null;
    } else {
      let a = (-h - p) / d;
      let b = (h - p) / d;
      if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a);
      t1 = Math.min(t1, b);
      if (t0 > t1) return null;
    }
  }
  return [t0, t1];
}

function egoClear(cam, px, py, pz, veh) {
  const belt = 0.6 * veh.H;
  const inset = cam.mount === 'exterior' ? 0.15 : 0;
  const hit = segBox(cam.x, cam.y, px, py, veh.L / 2 - inset, veh.W / 2 - inset);
  if (!hit) return true;
  const z = (t) => cam.z + (pz - cam.z) * t;
  if (cam.mount === 'cabin') return z(hit[1]) >= belt;
  if (cam.mount === 'roof') {
    if (z(hit[1]) < belt) return false;
    const roofFront = ROOF.front * (veh.L / 2);
    const roofRear = ROOF.rear * (veh.L / 2);
    const mid = (roofFront + roofRear) / 2;
    const half = (roofFront - roofRear) / 2;
    const roof = segBox(cam.x - mid, cam.y, px - mid, py, half - 0.02, veh.W / 2 - 0.02);
    return !roof || z(roof[1]) >= veh.H;
  }
  return z(hit[0]) >= veh.H && z(hit[1]) >= veh.H;
}

function objectBlocks(cam, px, py, pz, ob) {
  const r = -(ob.h || 0) * DEG;
  const c = Math.cos(r);
  const s = Math.sin(r);
  const loc = (x, y) => [(x - ob.x) * c - (y - ob.y) * s, (x - ob.x) * s + (y - ob.y) * c];
  const [ax, ay] = loc(cam.x, cam.y);
  const [bx, by] = loc(px, py);
  const k = CLASSES[ob.c];
  const hit = segBox(ax, ay, bx, by, k.l / 2, k.w / 2);
  if (!hit) return false;
  const z = (t) => cam.z + (pz - cam.z) * t;
  return z(hit[0]) < k.h || z(hit[1]) < k.h;
}

function inFrame(cam, px, py, pz) {
  const dx = px - cam.x;
  const dy = py - cam.y;
  const d = Math.hypot(dx, dy);
  if (d < 0.05) return false;
  if (angleDiff(Math.atan2(dy, dx), cam.yawR) > cam.fovR / 2) return false;
  const elevation = Math.atan2(pz - cam.z, d);
  return Math.abs(elevation + cam.pitchR) <= cam.vfovR / 2;
}

// Nearest distance at which this camera's frame still contains the road.
function groundMin(cam) {
  const steepest = cam.pitchR + cam.vfovR / 2;
  return steepest <= 0 ? Infinity : cam.z / Math.tan(Math.min(steepest, 80 * DEG));
}

const truncated = (cam, apparentM, d) => 2 * Math.atan2(apparentM / 2, d) > 0.85 * cam.fovR;

function singleError(cam, d, groundVisible, widthM, visibleFrac, cutOff, opts) {
  const f = focalPx(cam.fov, opts.inputPx);
  const sizeBased = cutOff
    ? Infinity
    : Math.hypot(opts.sizeErr * d, (d * d * opts.pxErr) / (f * widthM)) / Math.max(visibleFrac, 0.25);
  if (!groundVisible) return sizeBased;
  const pitch = opts.pitchErrDeg * DEG;
  const ground = Math.hypot((d * d * opts.pxErr) / (f * cam.z), (d * d * pitch) / cam.z, 0.15); // 0.15 m: tyres are never a sharp line
  return Math.min(ground, sizeBased);
}

function pairError(a, b, px, py, opts) {
  const ra = Math.atan2(py - a.y, px - a.x);
  const rb = Math.atan2(py - b.y, px - b.x);
  const gamma = Math.abs(Math.sin(ra - rb));
  if (gamma < 0.035) return Infinity;
  const d = (Math.hypot(px - a.x, py - a.y) + Math.hypot(px - b.x, py - b.y)) / 2;
  const sigma = opts.pxErr / Math.min(focalPx(a.fov, opts.inputPx), focalPx(b.fov, opts.inputPx));
  return Math.hypot((sigma * d) / gamma, 0.4); // 0.4 m: two views never box the same physical point
}

const placedWell = (err, d) => err <= Math.max(0.5, 0.15 * d);

// level: 0 unseen, 1 seen but position is a guess, 2 placed, 3 placed and cross-checked by a second view.
function evalPoint(cams, px, py, klass, veh, opts) {
  const k = CLASSES[klass];
  const seers = [];
  const pairable = [];
  let bestSingle = Infinity;
  for (const cam of cams) {
    const d = Math.hypot(px - cam.x, py - cam.y);
    if (d > detectRange(cam.fov, k.w, opts)) continue;
    const zTop = 0.7 * k.h;
    if (!inFrame(cam, px, py, zTop) || !egoClear(cam, px, py, zTop, veh)) continue;
    const ground = inFrame(cam, px, py, 0) && egoClear(cam, px, py, 0, veh);
    const cutOff = truncated(cam, k.l, d); // worst case: parallel traffic seen side-on
    seers.push(cam);
    if (!cutOff) pairable.push(cam);
    bestSingle = Math.min(bestSingle, singleError(cam, d, ground, k.w, 1, cutOff, opts));
  }
  if (!seers.length) return { level: 0, err: Infinity, n: 0 };
  let bestPair = Infinity;
  for (let i = 0; i < pairable.length; i++) {
    for (let j = i + 1; j < pairable.length; j++) bestPair = Math.min(bestPair, pairError(pairable[i], pairable[j], px, py, opts));
  }
  const d = Math.hypot(px, py);
  const err = Math.min(bestSingle, bestPair);
  let level = 1;
  if (placedWell(bestPair, d)) level = 3;
  else if (placedWell(bestSingle, d)) level = 2;
  return { level, err, n: seers.length };
}

function evalObject(ob, others, cams, veh, opts) {
  const k = CLASSES[ob.c];
  const seers = [];
  const pairable = [];
  let bestSingle = Infinity;
  for (const cam of cams) {
    const dist = Math.hypot(ob.x - cam.x, ob.y - cam.y);
    const toCam = Math.atan2(cam.y - ob.y, cam.x - ob.x);
    const perp = toCam + Math.PI / 2;
    const headingR = (ob.h || 0) * DEG;
    const across = Math.abs(Math.sin(toCam - headingR)) * k.l + Math.abs(Math.cos(toCam - headingR)) * k.w;
    const apparent = Math.min(across, k.l + k.w);
    let visible = 0;
    const offsets = [-0.4, -0.2, 0, 0.2, 0.4];
    for (const o of offsets) {
      const px = ob.x + Math.cos(perp) * o * apparent;
      const py = ob.y + Math.sin(perp) * o * apparent;
      const zTop = 0.7 * k.h;
      if (!inFrame(cam, px, py, zTop) || !egoClear(cam, px, py, zTop, veh)) continue;
      if (others.some((b) => objectBlocks(cam, px, py, zTop, b))) continue;
      visible += 1;
    }
    const frac = visible / offsets.length;
    if (frac < 0.39) continue;
    const seenWidth = apparent * frac;
    if (dist > detectRange(cam.fov, seenWidth, opts)) continue;
    const reach = Math.min(k.l, k.w) / 2;
    const gx = ob.x + Math.cos(toCam) * reach;
    const gy = ob.y + Math.sin(toCam) * reach;
    const ground = inFrame(cam, gx, gy, 0) && egoClear(cam, gx, gy, 0, veh)
      && !others.some((b) => objectBlocks(cam, gx, gy, 0, b));
    const cutOff = truncated(cam, apparent, dist);
    seers.push(cam);
    if (!cutOff && frac > 0.6) pairable.push(cam);
    bestSingle = Math.min(bestSingle, singleError(cam, dist, ground, k.w, frac, cutOff, opts));
  }
  const d = Math.hypot(ob.x, ob.y);
  if (!seers.length) return { state: 'hidden', err: Infinity, d, seers };
  let bestPair = Infinity;
  for (let i = 0; i < pairable.length; i++) {
    for (let j = i + 1; j < pairable.length; j++) bestPair = Math.min(bestPair, pairError(pairable[i], pairable[j], ob.x, ob.y, opts));
  }
  const err = Math.min(bestSingle, bestPair);
  return { state: placedWell(err, d) ? 'placed' : 'rough', err, d, seers };
}

function evalScenario(objects, cams, veh, opts) {
  return objects.map((ob) => evalObject(ob, objects.filter((o) => o !== ob), cams, veh, opts));
}

const RINGS = [
  { id: 'door', label: 'Door zone', from: 0.3, to: 3 },
  { id: 'near', label: 'Near', from: 3, to: 15 },
  { id: 'mid', label: 'Mid', from: 15, to: 40 },
  { id: 'far', label: 'Far', from: 40, to: 80 },
];

function ringScores(cams, klass, veh, opts) {
  const A = 180;
  const steps = 8;
  return RINGS.map((ring) => {
    let seen = 0;
    let placed = 0;
    let total = 0;
    for (let i = 0; i < A; i++) {
      const th = ((i + 0.5) / A) * 2 * Math.PI;
      const cx = Math.cos(th);
      const sy = Math.sin(th);
      const edge = Math.min(veh.L / 2 / Math.max(Math.abs(cx), 1e-6), veh.W / 2 / Math.max(Math.abs(sy), 1e-6));
      for (let j = 0; j < steps; j++) {
        const r = edge + ring.from + ((j + 0.5) / steps) * (ring.to - ring.from);
        const res = evalPoint(cams, cx * r, sy * r, klass, veh, opts);
        total += 1;
        if (res.level >= 1) seen += 1;
        if (res.level >= 2) placed += 1;
      }
    }
    return { ...ring, seen: seen / total, placed: placed / total };
  });
}

function fpsPerCamera(n, opts) {
  return (opts.chipFps * (640 / opts.inputPx) ** 2) / n;
}

// Cable route: straight to the nearest door seal or pillar, then along the sill to the box, plus the vertical drop.
function cableRun(cam, box, veh) {
  const sideY = Math.sign(cam.y || box.y || 1) * (veh.W / 2 - 0.1);
  const horizontal = Math.abs(cam.y - sideY) + Math.abs(cam.x - box.x) + Math.abs(sideY - box.y);
  return horizontal + Math.abs(cam.z - box.z) + 0.4;
}

const MODEL = {
  DEG, VEHICLES, CLASSES, DEFAULT_OPTS, SCENARIOS, RIGS, RINGS,
  placeCam, focalPx, detectRange, groundMin, inFrame, egoClear, evalPoint, evalObject, evalScenario, ringScores, fpsPerCamera, cableRun, segBox,
};
if (typeof module !== 'undefined') module.exports = MODEL;
