const M = require('./model.js');
const veh = M.VEHICLES.suv;
const opts = { ...M.DEFAULT_OPTS };
const pct = (v) => String(Math.round(v * 100)).padStart(3);

console.log('rig'.padEnd(12), 'cars: door/near/mid/far placed(seen) | bikes: door/near | scenes placed of total | fps per cam');
for (const rig of M.RIGS) {
  const cams = rig.cams.map((c) => M.placeCam(c, veh));
  const car = M.ringScores(cams, 'car', veh, opts);
  const bike = M.ringScores(cams, 'bike', veh, opts);
  let placed = 0;
  let total = 0;
  const per = [];
  for (const [id, s] of Object.entries(M.SCENARIOS)) {
    const res = M.evalScenario(s.objects, cams, veh, opts);
    const p = res.filter((r) => r.state === 'placed').length;
    const h = res.filter((r) => r.state === 'hidden').length;
    per.push(`${id}:${p}/${res.length}(h${h})`);
    placed += p;
    total += res.length;
  }
  console.log(
    rig.short.padEnd(12),
    car.map((r) => `${pct(r.placed)}(${pct(r.seen)})`).join(' '),
    '|',
    bike.slice(0, 2).map((r) => `${pct(r.placed)}(${pct(r.seen)})`).join(' '),
    '|',
    `${placed}/${total}`,
    per.join(' '),
    '|',
    M.fpsPerCamera(cams.length, opts).toFixed(0),
  );
}

console.log('\nlens table (640 input, car / bike / person detect range, m)');
for (const fov of [190, 150, 120, 90, 60, 45, 30]) {
  console.log(
    String(fov).padStart(4),
    M.focalPx(fov, 640).toFixed(0).padStart(5),
    M.detectRange(fov, 1.8, opts).toFixed(0).padStart(5),
    M.detectRange(fov, 0.75, opts).toFixed(0).padStart(5),
    M.detectRange(fov, 0.5, opts).toFixed(0).padStart(5),
  );
}
