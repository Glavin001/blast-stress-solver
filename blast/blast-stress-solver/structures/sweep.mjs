#!/usr/bin/env node
/**
 * Sweep one config parameter and tabulate what it does to the structure.
 *
 *   node structures/sweep.mjs parking-garage deckBeamDepth 1.1,1.3,1.5,1.7
 *   node structures/sweep.mjs parking-garage levels 3,4,5 --secs 30
 *
 * Two or more parameters at once, swept together rather than crossed:
 *
 *   node structures/sweep.mjs parking-garage deckBeamDepth=1.1,1.5 levelHeight=3.7,4.1
 *
 * WHY THIS EXISTS
 *
 * The parking garage's section was found by editing one constant, rebuilding,
 * running the stability audit, reading one number, and repeating -- twelve
 * times, at several minutes a turn, each turn producing a single data point
 * held in someone's head. Four of the six changes made things worse and it took
 * most of a session to see the shape that made obvious: deepening the beam
 * helped up to 1.5 m and hurt beyond it, because past that span the member is
 * carrying mostly itself.
 *
 * A U-shaped curve is invisible one point at a time and unmistakable in a
 * table. This prints the table.
 *
 * WHAT IT DOES
 *
 * For each value: build the pack with `--set`, then run the `structure-audit`
 * binary, which reports the same verdict the gate uses. Both steps are checked
 * -- a failed build aborts that row rather than silently auditing the previous
 * pack, which is the single most expensive mistake available here.
 */

import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const VIBE_LAND = process.env.VIBE_LAND_DIR ?? resolve(HERE, '../../../../vibe-land-2');

const argv = process.argv.slice(2);
if (argv.length < 2) {
  console.error(`usage:
  sweep.mjs <structure> <param> <v1,v2,...> [--secs N] [--keep]
  sweep.mjs <structure> <param>=<v1,v2,...> [<param2>=<...>] [--secs N] [--keep]

  --secs N   audit cap in seconds (default: the structure's own budget + 14)
  --keep     leave the last swept value built, instead of restoring defaults`);
  process.exit(2);
}

const structure = argv[0];
let secs = null;
let keep = false;
const rest = [];
for (let i = 1; i < argv.length; i++) {
  if (argv[i] === '--secs') { secs = argv[++i]; continue; }
  if (argv[i] === '--keep') { keep = true; continue; }
  rest.push(argv[i]);
}

// Either `param v1,v2` or `param=v1,v2 param2=v3,v4`.
const params = [];
if (rest.length === 2 && !rest[0].includes('=')) {
  params.push([rest[0], rest[1].split(',')]);
} else {
  for (const spec of rest) {
    const [k, v] = spec.split('=');
    if (!v) throw new Error(`expected param=v1,v2 — got "${spec}"`);
    params.push([k, v.split(',')]);
  }
}
const runs = Math.max(...params.map(([, vs]) => vs.length));
if (params.some(([, vs]) => vs.length !== runs && vs.length !== 1)) {
  throw new Error('every parameter needs the same number of values, or exactly one');
}

const build = (sets) => {
  const args = ['build.mjs', structure, '--emit-vibe-land', VIBE_LAND];
  for (const [k, v] of sets) args.push('--set', `${k}=${v}`);
  // Inherits stderr so a build failure is VISIBLE, and throws so the row is
  // skipped rather than audited against a stale pack.
  return execFileSync('node', args, { cwd: HERE, encoding: 'utf8' });
};

const audit = () => {
  const args = [structure];
  if (secs) args.push(secs);
  const out = execFileSync(resolve(VIBE_LAND, 'target/release/structure-audit'), args, {
    cwd: VIBE_LAND,
    encoding: 'utf8',
    // Capture stderr rather than inheriting it: the CUDA banner is printed on
    // every run and would bury the progress line it interleaves with.
    stdio: ['ignore', 'pipe', 'pipe'],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH
        ?? '/usr/local/cuda/lib64:/root/PhysX/physx/install/linux-clang/PhysX/bin/linux.x86_64/release',
    },
  });
  return JSON.parse(out.trim().split('\n').pop());
};

const rows = [];
for (let i = 0; i < runs; i++) {
  const sets = params.map(([k, vs]) => [k, vs[vs.length === 1 ? 0 : i]]);
  const label = sets.map(([k, v]) => `${k}=${v}`).join(' ');
  process.stderr.write(`  [${i + 1}/${runs}] ${label} ... `);
  let nodes = null;
  let bonds = null;
  try {
    const out = build(sets);
    const m = out.match(/(\d+) nodes\s+(\d+) bonds/);
    if (m) { nodes = +m[1]; bonds = +m[2]; }
    if (!/=> ok/.test(out)) {
      process.stderr.write('BUILD GATE FAILED\n');
      rows.push({ label, error: out.match(/FAIL\s+(.*)/)?.[1]?.slice(0, 60) ?? 'gate' });
      continue;
    }
  } catch (e) {
    process.stderr.write('BUILD ERROR\n');
    rows.push({ label, error: String(e.stderr ?? e.message).trim().split('\n').pop().slice(0, 60) });
    continue;
  }
  try {
    const r = audit();
    rows.push({ label, nodes, bonds, ...r });
    process.stderr.write(`${r.passes ? 'pass' : 'FAIL'} (${r.broke} broken)\n`);
  } catch (e) {
    process.stderr.write('AUDIT ERROR\n');
    rows.push({ label, nodes, bonds, error: String(e.stderr ?? e.message).trim().split('\n').pop().slice(0, 60) });
  }
}

const pad = (s, n) => String(s).padEnd(n);
const lpad = (s, n) => String(s).padStart(n);
const w = Math.max(10, ...rows.map((r) => r.label.length));
console.log('');
console.log(`  ${pad('setting', w)} ${lpad('bonds', 7)} ${lpad('broke', 6)} ${lpad('settled', 8)} ${lpad('peak', 6)} ${lpad('sag m', 7)} ${lpad('over', 6)}  verdict`);
for (const r of rows) {
  if (r.error) {
    console.log(`  ${pad(r.label, w)} ${lpad('-', 7)} ${lpad('-', 6)} ${lpad('-', 8)} ${lpad('-', 6)} ${lpad('-', 7)} ${lpad('-', 6)}  ${r.error}`);
    continue;
  }
  const settled = r.settles ? `${r.settled_at.toFixed(0)} s` : 'never';
  const sag = r.peak_sag > 0.01 ? `${r.peak_sag.toFixed(2)}` : '-';
  const verdict = r.passes
    ? 'stands'
    : r.breaks?.length
      ? `first: ${r.breaks[0].mode} ${r.breaks[0].class} y=${r.breaks[0].y}`
      : `worst class: ${r.classes?.[0]?.class ?? '?'}`;
  console.log(`  ${pad(r.label, w)} ${lpad(r.bonds ?? '-', 7)} ${lpad(r.broke, 6)} ${lpad(settled, 8)} ${lpad(r.late_peak.toFixed(2), 6)} ${lpad(sag, 7)} ${lpad(r.late_over.toFixed(0), 6)}  ${verdict}`);
}
console.log('');

// Leave the tree as it was found. A sweep that quietly leaves the last value
// built is a stale pack waiting to happen -- the exact failure the freshness
// check exists to catch, arriving through the tool meant to prevent it.
if (!keep) {
  process.stderr.write('  restoring defaults ... ');
  build([]);
  process.stderr.write('done\n');
}
