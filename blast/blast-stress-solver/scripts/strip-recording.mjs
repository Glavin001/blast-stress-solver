#!/usr/bin/env node
/**
 * Strip a session recording down to a **profile-only** bundle: keep every byte
 * the bottleneck analysis needs (per-frame `timing.*`, the scene-complexity
 * `columns.*`, and the sparse `resimLog`) and drop the heavy per-frame body
 * trajectory blob (`bodies`, ~95% of the file) plus the verbose topology event
 * stream (thousands of `migrate`/`destroy` deltas).
 *
 * The result is a *valid* `blast-sim-recording/v1` bundle — `decodeSimRecording`
 * and the analyzer consume it unchanged — but a few hundred KB instead of tens of
 * MB, so it is small enough to commit as a perf-regression fixture or attach to a
 * report. The body trajectory is intentionally gone, so trajectory queries
 * (`--body`, `--frame`) won't work on a stripped recording; that is by design.
 *
 * Usage:
 *   node scripts/strip-recording.mjs <in.sim.json[.gz]> <out.sim.json[.gz]>
 *
 * Output is gzipped when the path ends in `.gz`.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error('usage: node scripts/strip-recording.mjs <in.sim.json[.gz]> <out.sim.json[.gz]>');
  process.exit(1);
}

function load(path) {
  const raw = readFileSync(path);
  const text = path.endsWith('.gz') ? gunzipSync(raw).toString('utf8') : raw.toString('utf8');
  return JSON.parse(text);
}

const data = load(inPath);
if (data.schema !== 'blast-sim-recording/v1') {
  console.warn(`warning: unexpected schema "${data.schema}"`);
}

// An empty, valid encoded Float32 column — replaces the heavy `bodies` blob so the
// bundle still decodes (frame()/bodyInFrame() simply return no rows).
const EMPTY_F32 = { type: 'f32', length: 0, data: '' };
const EMPTY_I32 = { type: 'i32', length: 0, data: '' };
const EMPTY_F64 = { type: 'f64', length: 0, data: '' };

// Keep only the input/lifecycle events; drop the high-volume topology deltas
// (migrate/detach/destroy/bodyRemoved) which the profile analysis never reads.
const KEEP_EVENTS = new Set(['start', 'stop', 'projectile', 'force', 'gravity']);
const events = Array.isArray(data.events)
  ? data.events.filter((e) => KEEP_EVENTS.has(e.type))
  : [];

const slim = {
  schema: data.schema,
  generatedAt: data.generatedAt,
  environment: data.environment,
  durationFrames: data.durationFrames,
  durationSeconds: data.durationSeconds,
  meta: data.meta,
  // Scenario topology is large and unused by profile analysis — keep just a summary.
  scenarioSummary: {
    nodeCount: data.scenario?.nodes?.length ?? 0,
    bondCount: data.scenario?.bonds?.length ?? 0,
  },
  initialBodyByNode: EMPTY_I32,
  nodeIndices: EMPTY_I32,
  bodyStride: data.bodyStride,
  bodyLayout: data.bodyLayout,
  handleTable: EMPTY_F64,
  columns: data.columns, // 8 small scene-complexity columns (the analysis x-axes)
  bodies: EMPTY_F32, // ← the big drop: per-frame body trajectory removed
  events,
  timing: data.timing, // ← the analysis payload: per-frame phase timers
  resimLog: data.resimLog ?? [],
  // A breadcrumb so readers know this is a derived, lossy bundle.
  strippedProfileOnly: true,
};

const json = JSON.stringify(slim);
const out = outPath.endsWith('.gz') ? gzipSync(json, { level: 9 }) : Buffer.from(json);
writeFileSync(outPath, out);

const inBytes = readFileSync(inPath).length;
console.log(
  `stripped ${inPath} (${(inBytes / 1024 / 1024).toFixed(1)} MB) → ` +
    `${outPath} (${(out.length / 1024).toFixed(0)} KB)  ` +
    `[${data.durationFrames} frames, ${events.length} input events kept]`,
);
