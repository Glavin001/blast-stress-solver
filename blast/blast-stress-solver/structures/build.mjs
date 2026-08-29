#!/usr/bin/env node
/**
 * Generate the authored structures as ScenePacks.
 *
 *   node structures/build.mjs                        # all of them
 *   node structures/build.mjs algedra-tower          # just one
 *   node structures/build.mjs --emit-vibe-land /path/to/vibe-land-2
 *
 * Packs land in this repo's own assets/scenes, and additionally in
 * vibe-land-2's destruction/assets/scenes when --emit-vibe-land is given —
 * the same cross-repo hand-off scripts/export-rust-scenes.mjs already makes.
 * Once there they load via VIBE_CITY_SCENE=<name>.json.
 *
 * Every pack is verified before it is written. Generation is seeded, so the
 * same source produces byte-identical output.
 */
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildAlgedra } from './algedra-tower.mjs';
import { buildHouse1 } from './house-1story.mjs';
import { buildHouse2 } from './house-2story.mjs';
import { buildVillaSavoye } from './villa-savoye.mjs';
import { buildPark432 } from './park-432.mjs';
import { buildParkingGarage } from './parking-garage.mjs';
import { buildPetronas } from './petronas.mjs';
import { buildMinasTirith } from './minas-tirith.mjs';
import { buildNeighbourhood, buildSkyline } from './neighbourhood.mjs';
import {
  buildRigCantilever, buildRigColumn, buildRigGarage, buildRigPane,
  buildRigPortal, buildRigToppled, buildRigWall,
} from './rigs.mjs';
import { standalone, row } from './components.mjs';
import { shardHistogram } from './lib/fracture.mjs';
import { verifyPack } from './verify.mjs';
import { applyAutoBonds } from './lib/autobond.mjs';
import { cullSliverBonds } from './lib/sliver.mjs';
import { buildShapeLibrary } from './lib/colliders.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const STRUCTURES = {
  'algedra-tower': buildAlgedra,
  'house-1story': buildHouse1,
  'house-2story': buildHouse2,
  'villa-savoye': buildVillaSavoye,
  'park-432': buildPark432,
  'parking-garage': buildParkingGarage,
  petronas: buildPetronas,
  'minas-tirith': buildMinasTirith,
  // All three in one scene, for looking at or playing in.
  neighbourhood: buildNeighbourhood,
  skyline: buildSkyline,
  // Test rigs: small structures that each answer one structural question, so
  // a scenario suite can run in minutes instead of hours. See rigs.mjs.
  'rig-column': buildRigColumn,
  'rig-portal': buildRigPortal,
  'rig-cantilever': buildRigCantilever,
  'rig-garage': buildRigGarage,
  'rig-pane': buildRigPane,
  'rig-wall': buildRigWall,
  'rig-toppled': buildRigToppled,
  // Components: one bay, and the same bay composed, so "does it compose" is a
  // measurement rather than an assumption. See components.mjs.
  'comp-frame-bay': () => standalone('frame-bay'),
  'comp-frame-bay-x2': () => row('frame-bay', 2),
  'comp-frame-bay-x4': () => row('frame-bay', 4),
  'comp-frame-bay-x8': () => row('frame-bay', 8),
  'comp-wall-bay': () => standalone('wall-bay'),
  'comp-wall-bay-x2': () => row('wall-bay', 2),
  'comp-wall-bay-x4': () => row('wall-bay', 4),
  'comp-wall-bay-x8': () => row('wall-bay', 8),
};

const argv = process.argv.slice(2);
const names = [];
let vibeLand = null;
let autobond = true;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--emit-vibe-land') { vibeLand = argv[++i]; continue; }
  if (argv[i] === '--no-autobond') { autobond = false; continue; }
  names.push(argv[i]);
}
/**
 * Built only when asked for by name.
 *
 * `skyline` is every authored structure in one scene: 56,509 chunks and
 * 1.5 million hull points. The client builds draw data for every chunk in a
 * scene up front and the GPU upload happens lazily, on the first frame a mesh
 * is actually rendered -- so the whole cost arrives in one frame the moment
 * the buildings come into view, which is enough to have Safari kill the tab on
 * an iPhone. It survives loading, and it survives looking at empty sky; it
 * dies when you turn around.
 *
 * It is still a useful scene for desktop work and for anything that wants
 * every building at once, so it stays buildable -- `node build.mjs skyline` --
 * and simply stops being something a plain rebuild hands to a phone.
 */
const OPT_IN_ONLY = new Set(['skyline']);

const selected = names.length
  ? names
  : Object.keys(STRUCTURES).filter((name) => !OPT_IN_ONLY.has(name));
for (const n of selected) {
  if (!STRUCTURES[n]) {
    console.error(`unknown structure "${n}" (have: ${Object.keys(STRUCTURES).join(', ')})`);
    process.exit(2);
  }
}

const outDirs = [path.resolve(__dirname, '../../blast-stress-demo-rs/assets/scenes')];
if (vibeLand) outDirs.push(path.resolve(vibeLand, 'destruction/assets/scenes'));

let failed = 0;
for (const name of selected) {
  const started = process.hrtime.bigint();
  const { pack: result, builder } = STRUCTURES[name]();
  const ms = Number(process.hrtime.bigint() - started) / 1e6;
  const s = result.scenario;

  const types = {};
  for (const t of s.nodeTypes) types[t] = (types[t] ?? 0) + 1;
  const mats = {};
  for (const m of s.nodeMaterials) mats[m] = (mats[m] ?? 0) + 1;
  const hulls = s.nodeColliders.filter((c) => c.kind === 'convex_hull').length;

  console.log(`\n=== ${name} — ${result.title}`);
  console.log(`  ${s.nodes.length} nodes  ${s.bonds.length} bonds  ` +
    `${hulls} hulls / ${s.nodes.length - hulls} cuboids  (${ms.toFixed(0)} ms)`);
  console.log(`  types     ${Object.entries(types).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.log(`  materials ${Object.entries(mats).map(([k, v]) => `${k}=${v}`).join(' ')}`);
  console.log(shardHistogram(builder.shardStats));
  if (builder.penetratingContacts) {
    console.log(`  note: ${builder.penetratingContacts} contact(s) formed through slight overlap ` +
      `rather than a clean face (a pitched roof on a square plate always makes a few)`);
  }

  if (autobond) {
    // Re-measure contact with NvBlast's own bond generator before verifying, so
    // the statics check reads the areas the pack will actually ship with.
    const ab = await applyAutoBonds(result);
    console.log(`  autobond  ${ab.remeasured} re-measured (${ab.shrunk} shrunk)  ` +
      `${ab.kept} flush kept  ${ab.dropped} dropped  ${ab.added} added  ` +
      `-> ${s.bonds.length} bonds, ${ab.areaBefore.toFixed(0)} -> ${ab.areaAfter.toFixed(0)} m^2`);
  }

  // After auto-bonding, because that is when contact areas are final: it
  // re-measures every one and adds contacts the closed-form pass missed, so
  // culling before it just gets the slivers put back.
  const cull = cullSliverBonds(result);
  if (cull.dropped > 0) {
    console.log(`  slivers   ${cull.dropped} contacts below `
      + `${(100 * 0.02).toFixed(0)}% of their smaller chunk's face dropped `
      + `-> ${cull.kept} bonds`
      + (cull.rescued > 0 ? `  (${cull.rescued} kept to avoid orphaning a chunk)` : ''));
  }

  // After bonding, before verifying: the validators resolve references, so the
  // pack they check is the pack that ships.
  const lib = buildShapeLibrary(s);
  if (lib.library > 0) {
    console.log(`  shapes    ${lib.hulls} hulls -> ${lib.library} library entries + ` +
      `${lib.inline} one-offs  (${(lib.hulls / (lib.library + lib.inline)).toFixed(1)}x reuse)`);
  }

  const report = verifyPack(result, `  verify ${name}`);
  if (!report.ok) { failed++; continue; }

  const json = JSON.stringify(result);
  for (const dir of outDirs) {
    await mkdir(dir, { recursive: true });
    const file = path.join(dir, `${name}.json`);
    await writeFile(file, json);
    console.log(`  wrote ${file}  (${(json.length / 1e6).toFixed(2)} MB)`);
  }
}

if (failed) {
  console.error(`\n${failed} structure(s) failed verification and were NOT written`);
  process.exit(1);
}
