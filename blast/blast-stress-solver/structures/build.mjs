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
import { buildNeighbourhood, buildSkyline } from './neighbourhood.mjs';
import { shardHistogram } from './lib/fracture.mjs';
import { verifyPack } from './verify.mjs';
import { applyAutoBonds } from './lib/autobond.mjs';
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
  // All three in one scene, for looking at or playing in.
  neighbourhood: buildNeighbourhood,
  skyline: buildSkyline,
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
const selected = names.length ? names : Object.keys(STRUCTURES);
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
