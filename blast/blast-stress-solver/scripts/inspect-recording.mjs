#!/usr/bin/env node
/**
 * Inspect / query a session recording produced by the in-app ⏺ Session recorder
 * (the gzipped `*.sim.json.gz` bundle, or a plain `*.sim.json`).
 *
 * Demonstrates how to consume the recording format for offline analysis: it
 * gunzips, parses, and decodes the base64 columnar trace back into typed arrays
 * (the same logic as the package's exported `decodeSimRecording`, inlined here so
 * the tool is fully self-contained — no build / no WASM needed), then prints a
 * summary and answers a few example questions.
 *
 * Usage:
 *   node scripts/inspect-recording.mjs <file.sim.json[.gz]> [options]
 *
 * Options:
 *   --body <handle>        Print the trajectory (pos + speed) of one rigid body.
 *   --frame <i>            Dump every body's row for frame i.
 *   --events [type]        List timeline events (optionally filtered by type:
 *                          projectile|force|gravity|migrate|detach|destroy|bodyRemoved).
 *   --range <a> <b>        Limit --body / --events output to frames [a, b].
 *   --perf                 Per-phase timing breakdown ("where did every ms go") +
 *                          the worst (slowest) frames and their dominant phase.
 *
 * Examples:
 *   node scripts/inspect-recording.mjs rec.sim.json.gz
 *   node scripts/inspect-recording.mjs rec.sim.json.gz --events projectile
 *   node scripts/inspect-recording.mjs rec.sim.json.gz --body 42 --range 90 140
 */
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

// ── Inlined decoder (mirrors `decodeSimRecording` from the package) ───────────
function decodeTyped(enc) {
  const bytes = Buffer.from(enc.data, 'base64');
  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  switch (enc.type) {
    case 'f64': return new Float64Array(ab, 0, enc.length);
    case 'f32': return new Float32Array(ab, 0, enc.length);
    case 'u32': return new Uint32Array(ab, 0, enc.length);
    case 'i32': return new Int32Array(ab, 0, enc.length);
    default: throw new Error(`unknown encoded type: ${enc.type}`);
  }
}

function decodeSimRecording(data) {
  const columns = {
    simTime: decodeTyped(data.columns.simTime),
    dt: decodeTyped(data.columns.dt),
    bodyCount: decodeTyped(data.columns.bodyCount),
    activeBonds: decodeTyped(data.columns.activeBonds),
    rigidBodies: decodeTyped(data.columns.rigidBodies),
    projectiles: decodeTyped(data.columns.projectiles),
  };
  const bodies = decodeTyped(data.bodies);
  const stride = data.bodyStride;
  const n = columns.bodyCount.length;
  const frameBodyOffset = new Uint32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    frameBodyOffset[i] = acc;
    acc += columns.bodyCount[i] * stride;
  }
  const frame = (i) => {
    if (i < 0 || i >= n) return new Float32Array(0);
    const s = frameBodyOffset[i];
    return bodies.subarray(s, s + columns.bodyCount[i] * stride);
  };
  const bodyInFrame = (i, handle) => {
    const rows = frame(i);
    for (let off = 0; off < rows.length; off += stride) {
      if (rows[off] === handle) return rows.subarray(off, off + stride);
    }
    return null;
  };
  const timing = {};
  if (data.timing) {
    for (const f of data.timing.fields) {
      const enc = data.timing.columns[f];
      if (enc) timing[f] = decodeTyped(enc);
    }
  }

  return {
    durationFrames: data.durationFrames,
    durationSeconds: data.durationSeconds,
    bodyStride: stride,
    bodyLayout: data.bodyLayout,
    columns,
    bodies,
    frameBodyOffset,
    events: data.events,
    timing,
    resimLog: data.resimLog ?? [],
    handleTable: data.handleTable ? decodeTyped(data.handleTable) : new Float64Array(0),
    frame,
    bodyInFrame,
  };
}

// Leaf phases (≈ mutually exclusive — they sum to ~totalMs). Wrapper totals
// (initialPassMs/resimMs) and counts are intentionally excluded from the sum.
const LEAF_TIMERS = [
  'rapierStepMs', 'solverUpdateMs', 'contactDrainMs', 'externalForceMs', 'preStepSweepMs',
  'fractureMs', 'splitPlannerMs', 'splitQueueMs', 'bodyCreateMs', 'colliderRebuildMs',
  'rebuildColliderMapMs', 'cleanupDisabledMs', 'snapshotCaptureMs', 'snapshotRestoreMs',
  'damageTickMs', 'damageReplayMs', 'damagePreviewMs', 'damageRestoreMs', 'damageSnapshotMs',
  'damagePreDestroyMs', 'damageFlushMs', 'spawnMs', 'projectileCleanupMs',
];

const argv = process.argv.slice(2);
if (argv.length === 0 || argv[0].startsWith('--')) {
  console.error('usage: node scripts/inspect-recording.mjs <file.sim.json[.gz]> [--body h] [--frame i] [--events [type]] [--range a b]');
  process.exit(1);
}

const file = argv[0];
const opt = (name) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : undefined;
};
const has = (name) => argv.includes(name);

function loadRecording(path) {
  const raw = readFileSync(path);
  const text = path.endsWith('.gz') ? gunzipSync(raw).toString('utf8') : raw.toString('utf8');
  return JSON.parse(text);
}

const data = loadRecording(file);
if (data.schema !== 'blast-sim-recording/v1') {
  console.warn(`warning: unexpected schema "${data.schema}"`);
}
const dec = decodeSimRecording(data);

// ── Summary ───────────────────────────────────────────────────────────────────
const env = data.environment ?? {};
console.log('━━━ Session recording ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`file        : ${file}`);
console.log(`generatedAt : ${data.generatedAt}`);
console.log(`page        : ${env.page ?? '?'}   (${env.userAgent ?? 'ua unknown'})`);
console.log(`frames      : ${dec.durationFrames}   duration: ${dec.durationSeconds.toFixed(2)} s`);
console.log(`meta        : ${JSON.stringify(data.meta ?? {})}`);

const bc = dec.columns.bodyCount;
const rb = dec.columns.rigidBodies;
const ab = dec.columns.activeBonds;
const peakBodies = bc.reduce((m, v) => Math.max(m, v), 0);
console.log(`bodies/frame: min ${Math.min(...bc)} · max ${peakBodies}`);
console.log(`rigidBodies : start ${rb[0]} → end ${rb[rb.length - 1]} (peak ${rb.reduce((m, v) => Math.max(m, v), 0)})`);
console.log(`activeBonds : start ${ab[0]} → end ${ab[ab.length - 1]} (Δ ${ab[ab.length - 1] - ab[0]})`);

// Event histogram.
const hist = {};
for (const e of dec.events) hist[e.type] = (hist[e.type] ?? 0) + 1;
console.log(`events      : ${JSON.stringify(hist)}`);
if (data.scenario) {
  const n = data.scenario.nodes?.length ?? '?';
  const b = data.scenario.bonds?.length ?? '?';
  console.log(`scenario    : ${n} nodes · ${b} bonds`);
}
const hasTiming = dec.timing && dec.timing.totalMs && dec.timing.totalMs.length === dec.durationFrames;
console.log(`timing      : ${hasTiming ? `full-session (${dec.timing.totalMs.length} frames) — run with --perf` : 'none'}`);

// ── Range ──────────────────────────────────────────────────────────────────────
const range = has('--range')
  ? [Number(argv[argv.indexOf('--range') + 1]), Number(argv[argv.indexOf('--range') + 2])]
  : [0, dec.durationFrames - 1];

// ── --perf (where did every millisecond go) ─────────────────────────────────────
if (has('--perf')) {
  if (!hasTiming) {
    console.log('\n(no timing stream in this recording)');
  } else {
    const t = dec.timing;
    const nf = t.totalMs.length;
    const [a, b] = range;
    const sum = (arr) => { let s = 0; for (let i = a; i <= b && i < nf; i++) s += arr[i] || 0; return s; };
    const totalAll = sum(t.totalMs);
    const cat = LEAF_TIMERS.filter((k) => t[k]).map((k) => ({ k, ms: sum(t[k]) }))
      .filter((c) => c.ms > 1e-6).sort((x, y) => y.ms - x.ms);
    const leafSum = cat.reduce((s, c) => s + c.ms, 0);
    const other = Math.max(0, totalAll - leafSum);
    const fms = (v) => v.toFixed(1).padStart(9);
    const pct = (v) => `${(totalAll > 0 ? (100 * v) / totalAll : 0).toFixed(1)}%`.padStart(6);
    console.log(`\n━━━ Performance breakdown · frames ${a}..${Math.min(b, nf - 1)} ━━━`);
    console.log(`total sim time: ${totalAll.toFixed(1)} ms over ${b - a + 1} frames · mean ${(totalAll / (b - a + 1)).toFixed(2)} ms/frame`);
    console.log('\n  phase                    Σ ms     %total   mean ms/frame');
    for (const c of cat)
      console.log(`  ${c.k.padEnd(22)} ${fms(c.ms)}  ${pct(c.ms)}   ${(c.ms / (b - a + 1)).toFixed(3)}`);
    console.log(`  ${'(unaccounted/other)'.padEnd(22)} ${fms(other)}  ${pct(other)}`);
    console.log(`  accounted by leaf phases: ${(100 * leafSum / totalAll).toFixed(1)}%`);

    // ── Solver phase breakdown (sub-phases of solverUpdateMs) ───────────────────
    // solverUpdateMs is a wrapper: JS gravity fill + JS contact/splash injection
    // + the WASM CGNR solve. Splitting it tells us whether the next perf effort
    // belongs in JS (our injection code) or the vendored solver.
    const solverTotal = sum(t.solverUpdateMs || []);
    if (solverTotal > 1e-6) {
      const subDefs = [
        ['gravity inject (JS)', 'solverGravityInjectMs'],
        ['contact inject (JS)', 'solverContactInjectMs'],
        ['CGNR solve   (WASM)', 'solverSolveMs'],
      ];
      const haveSub = subDefs.some(([, k]) => t[k]);
      console.log(`\n  ── Solver phase breakdown (Σ solverUpdateMs ${solverTotal.toFixed(1)} ms) ──`);
      if (!haveSub) {
        console.log('  (sub-phase timers not present — recording predates the solver breakdown)');
      } else {
        const spct = (v) => `${(solverTotal > 0 ? (100 * v) / solverTotal : 0).toFixed(1)}%`.padStart(6);
        let subSum = 0;
        for (const [label, k] of subDefs) {
          const ms = sum(t[k] || []);
          subSum += ms;
          console.log(`  ${label.padEnd(22)} ${fms(ms)}  ${spct(ms)} of solver   ${(ms / (b - a + 1)).toFixed(3)} ms/frame`);
        }
        const subOther = Math.max(0, solverTotal - subSum);
        console.log(`  ${'(predicate/overhead)'.padEnd(22)} ${fms(subOther)}  ${spct(subOther)} of solver`);
      }
    }

    // Slowest frames + their dominant phase (the spikes worth investigating).
    const order = [...Array(nf).keys()].filter((i) => i >= a && i <= b).sort((i, j) => t.totalMs[j] - t.totalMs[i]).slice(0, 10);
    console.log('\n  slowest frames:  frame  totalMs   dominant phase           resimPasses  bodies');
    for (const i of order) {
      let dk = 'other', dv = -1;
      for (const k of LEAF_TIMERS) if (t[k] && t[k][i] > dv) { dv = t[k][i]; dk = k; }
      const resim = t.resimPasses ? t.resimPasses[i] : 0;
      const bodies = t.rigidBodies ? t.rigidBodies[i] : (dec.columns.rigidBodies[i] || 0);
      console.log(`                   ${String(i).padStart(5)}  ${t.totalMs[i].toFixed(2).padStart(6)}   ${dk.padEnd(20)} ${dv.toFixed(2).padStart(7)}   ${String(resim).padStart(6)}    ${bodies}`);
    }

    // ── Resimulation cost ──────────────────────────────────────────────────────
    const resimMs = t.resimMs, initMs = t.initialPassMs, resimN = t.resimPasses;
    const inRange = (i) => i >= a && i <= b;
    if (resimMs) {
      let totResim = 0, totInit = 0, resimFrames = 0, totPasses = 0;
      for (let i = a; i <= b && i < nf; i++) {
        totResim += resimMs[i] || 0; totInit += (initMs ? initMs[i] : 0) || 0;
        if ((resimN ? resimN[i] : 0) > 0) { resimFrames++; totPasses += resimN[i]; }
      }
      console.log(`\n  ── Resimulation ──`);
      console.log(`  resim cost: ${totResim.toFixed(1)} ms total (${pct(totResim)} of frame time) · base-pass ${totInit.toFixed(1)} ms`);
      console.log(`  resim fired on ${resimFrames} frames · ${totPasses} resim passes total · ${resimFrames ? (totResim / resimFrames).toFixed(2) : 0} ms per resim frame`);

      // Per-pass detail from the sparse resimLog, worst resim frames first.
      const log = (dec.resimLog || []).filter((e) => inRange(e.f)).sort((x, y) => passSum(y) - passSum(x)).slice(0, 6);
      if (log.length) {
        console.log(`\n  worst resim frames (per-pass: index·totalMs [solver/fracture/bodyCreate]·reasons):`);
        for (const e of log) {
          const total = passSum(e).toFixed(2);
          console.log(`    f${String(e.f).padStart(4)}  Σ${total}ms over ${e.passes.length} passes:`);
          for (const p of e.passes)
            console.log(`        pass ${p.index}: ${p.totalMs.toFixed(2)}ms  [solver ${p.solverMs.toFixed(2)} · fracture ${p.fractureMs.toFixed(2)} · bodyCreate ${p.bodyCreateMs.toFixed(2)}]  ${JSON.stringify(p.reasons)}`);
        }
      } else if (totPasses > 0) {
        console.log(`  (per-pass detail not present — recording predates resimLog)`);
      }
    }
  }
}
function passSum(e) { return e.passes.reduce((s, p) => s + p.totalMs, 0); }

// ── --events ────────────────────────────────────────────────────────────────────
if (has('--events')) {
  const type = opt('--events');
  const filterType = type && !type.startsWith('--') ? type : null;
  console.log(`\n━━━ Events ${filterType ? `(type=${filterType})` : ''} frames ${range[0]}..${range[1]} ━━━`);
  for (const e of dec.events) {
    if (e.f < range[0] || e.f > range[1]) continue;
    if (filterType && e.type !== filterType) continue;
    console.log(`  f${String(e.f).padStart(5)} t=${e.t.toFixed(3)}s  ${e.type.padEnd(11)} ${JSON.stringify(stripFT(e))}`);
  }
}

// ── --body <handle> ──────────────────────────────────────────────────────────────
if (has('--body')) {
  const handle = Number(opt('--body'));
  const stride = dec.bodyStride;
  const raw = dec.handleTable && dec.handleTable[handle] != null ? dec.handleTable[handle] : '?';
  console.log(`\n━━━ Body id ${handle} (raw rapier handle ${raw}) — trajectory frames ${range[0]}..${range[1]} ━━━`);
  console.log('  frame    t(s)        pos(x,y,z)                 |linvel|  |angvel|');
  let prev = null;
  for (let f = range[0]; f <= range[1] && f < dec.durationFrames; f += 1) {
    const row = dec.bodyInFrame(f, handle);
    if (!row) {
      if (prev) console.log(`  f${String(f).padStart(5)}  (body absent from this frame)`);
      prev = null;
      continue;
    }
    const [, px, py, pz, , , , , lvx, lvy, lvz, avx, avy, avz] = row;
    const speed = Math.hypot(lvx, lvy, lvz);
    const aspeed = Math.hypot(avx, avy, avz);
    // Flag suspicious jumps (a teleport/jitter signal) vs the previous frame.
    let flag = '';
    if (prev) {
      const jump = Math.hypot(px - prev[0], py - prev[1], pz - prev[2]);
      const expected = speed * (dec.columns.dt[f] || 1 / 60);
      if (jump > Math.max(0.25, expected * 4)) flag = `  ⚠ jump ${jump.toFixed(3)}m (expected ~${expected.toFixed(3)})`;
    }
    console.log(
      `  f${String(f).padStart(5)}  ${dec.columns.simTime[f].toFixed(3).padStart(7)}  ` +
        `(${px.toFixed(2)}, ${py.toFixed(2)}, ${pz.toFixed(2)})`.padEnd(28) +
        `  ${speed.toFixed(3).padStart(7)}  ${aspeed.toFixed(3).padStart(7)}${flag}`,
    );
    prev = [px, py, pz];
  }
}

// ── --frame <i> ──────────────────────────────────────────────────────────────────
if (has('--frame')) {
  const i = Number(opt('--frame'));
  const rows = dec.frame(i);
  const stride = dec.bodyStride;
  console.log(`\n━━━ Frame ${i} — ${rows.length / stride} bodies (t=${dec.columns.simTime[i]?.toFixed(3)}s) ━━━`);
  console.log(`  ${dec.bodyLayout.join('  ')}`);
  for (let off = 0; off < rows.length; off += stride) {
    const r = Array.from(rows.subarray(off, off + stride)).map((v, k) => (k === 0 ? String(v) : v.toFixed(3)));
    console.log('  ' + r.join('  '));
  }
}

function stripFT(e) {
  const { f, t, type, ...rest } = e;
  return rest;
}
