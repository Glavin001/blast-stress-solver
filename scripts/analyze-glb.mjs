#!/usr/bin/env node
/**
 * analyze-glb.mjs — GLB decomposition analyzer / manifest generator.
 *
 * A dependency-free CLI that inspects a `.glb` model and helps you turn it into a
 * destructible scenario. For every mesh part it reports world position, size,
 * triangle count and material, then auto-classifies the part into a structural
 * "role" (frame / wheel / panel / cargo / accessory / ground) using name,
 * material and geometry heuristics. Roles drive the bond-strength hierarchy in
 * the destructible-vehicle demo (frame is strongest, cargo falls off first).
 *
 * It also writes a JSON *decomposition manifest* (part name -> role) that you can
 * hand-edit when the heuristics get a part wrong, then feed back into the demo.
 *
 * Usage:
 *   node scripts/analyze-glb.mjs <model.glb> [--json out.json] [--quiet]
 *
 * The heuristics here are intentionally mirrored by classifyVehiclePart() in
 * blast/js_stress_example/glb-vehicle.ts so the CLI and the browser demo agree.
 */
import fs from 'node:fs';
import path from 'node:path';

// ── arg parsing ──────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith('--'));
const flags = new Map();
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    const key = args[i].slice(2);
    const val = args[i + 1] && !args[i + 1].startsWith('--') ? args[++i] : 'true';
    flags.set(key, val);
  }
}
const inputPath = positional[0];
const quiet = flags.get('quiet') === 'true';
if (!inputPath) {
  console.error('Usage: node scripts/analyze-glb.mjs <model.glb> [--json out.json] [--quiet]');
  process.exit(1);
}

// ── GLB parsing (binary glTF, no deps) ───────────────────────────────────────
function parseGlb(file) {
  const buf = fs.readFileSync(file);
  const magic = buf.readUInt32LE(0);
  if (magic !== 0x46546c67) throw new Error('Not a GLB file (bad magic)');
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(new TextDecoder().decode(buf.subarray(20, 20 + jsonLen)));
  return { json, byteLength: buf.length, jsonLen };
}

// column-major glTF mat4 * point
function transformPoint(m, p) {
  const [x, y, z] = p;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}
const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Extract one record per mesh-bearing node, with world-space bbox + material. */
function extractParts(json) {
  const nodes = json.nodes ?? [];
  const meshes = json.meshes ?? [];
  const accessors = json.accessors ?? [];
  const materials = json.materials ?? [];
  const parts = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.mesh === undefined) continue;
    const m = n.matrix || IDENT;
    const mesh = meshes[n.mesh];
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];
    let tris = 0;
    const mats = [];
    for (const prim of mesh.primitives ?? []) {
      const acc = accessors[prim.attributes?.POSITION];
      if (acc?.min && acc?.max) {
        for (let k = 0; k < 3; k++) {
          lo[k] = Math.min(lo[k], acc.min[k]);
          hi[k] = Math.max(hi[k], acc.max[k]);
        }
      }
      if (prim.indices !== undefined && accessors[prim.indices]) {
        tris += Math.floor(accessors[prim.indices].count / 3);
      } else if (acc) {
        tris += Math.floor(acc.count / 3);
      }
      if (prim.material !== undefined) mats.push(materials[prim.material]?.name ?? `mat${prim.material}`);
    }
    if (!Number.isFinite(lo[0])) continue;
    // transform 8 corners to world space
    let wlo = [Infinity, Infinity, Infinity];
    let whi = [-Infinity, -Infinity, -Infinity];
    for (let cx = 0; cx < 2; cx++)
      for (let cy = 0; cy < 2; cy++)
        for (let cz = 0; cz < 2; cz++) {
          const c = transformPoint(m, [cx ? hi[0] : lo[0], cy ? hi[1] : lo[1], cz ? hi[2] : lo[2]]);
          for (let k = 0; k < 3; k++) {
            wlo[k] = Math.min(wlo[k], c[k]);
            whi[k] = Math.max(whi[k], c[k]);
          }
        }
    const size = [whi[0] - wlo[0], whi[1] - wlo[1], whi[2] - wlo[2]];
    const center = [(wlo[0] + whi[0]) / 2, (wlo[1] + whi[1]) / 2, (wlo[2] + whi[2]) / 2];
    parts.push({
      node: i,
      name: n.name || `node_${i}`,
      tris,
      material: [...new Set(mats)].join(',') || 'none',
      center,
      size,
      wlo,
      whi,
    });
  }
  return parts;
}

// ── classification heuristics (kept in sync with glb-vehicle.ts) ─────────────
//
// Returns one of: frame | wheel | panel | cargo | accessory | ground
// `bounds` is the whole-vehicle bbox (excluding ground) used for relative cues.
export function classifyPart(part, bounds) {
  const name = part.name.toLowerCase();
  const sx = part.size[0];
  const sy = part.size[1];
  const sz = part.size[2];
  const maxDim = Math.max(sx, sy, sz);
  const minDim = Math.min(sx, sy, sz);
  const carLen = Math.max(bounds.size[0], bounds.size[2]) || 1;
  const carHeight = bounds.size[1] || 1;
  // normalized height of the part centre within the vehicle (0 = floor, 1 = roof)
  const relY = (part.center[1] - bounds.lo[1]) / carHeight;
  const spans = maxDim / carLen; // how much of the vehicle length this part covers

  // Backdrop / floor plane from the original scene (e.g. a 12000-unit plane).
  if (maxDim > carLen * 3) return 'ground';

  // 1) Name keywords — the strongest signal when an artist labelled parts.
  if (/wheel|tire|tyre|\brim\b/.test(name)) return 'wheel';
  if (/cage|chassis|\bframe\b|\bbody|rollbar|roll_?cage|rollcage/.test(name)) return 'frame';
  if (/door|hood|bonnet|fender|bumper|panel|windshield|windscreen|glass|window/.test(name)) return 'panel';
  if (/seat|interior|dash|steer/.test(name)) return 'panel';
  if (/engine|motor|axle|suspension|exhaust|drivetrain|gearbox/.test(name)) return 'frame';
  // Quixel Megascans + obvious props are strapped-on cargo.
  if (/^aset_|barrel|drum|crate|\bbox\b|\blog\b|rock|tarp|jerry|canister|\bcan\b|bag|sack|cargo|container|plastic|wood|fabric|cloth/.test(name)) {
    return 'cargo';
  }
  if (/chain|rope|cable|hook|bucket|pipe|wire|strap|antenna/.test(name)) return 'accessory';

  // 2) Geometry/material fallback for generically-named parts
  //    (Circle/Cube/Plane/Cylinder/NurbsPath/p_low/...).
  // Big parts that span most of the vehicle and sit in the structural mid-body
  // are the skeleton (chassis tubes, underbody, central spine).
  if (spans > 0.6) return 'frame';
  // Compact, low-sitting, roughly square footprint => likely a wheel/hub.
  if (relY < 0.38 && minDim > maxDim * 0.4 && spans < 0.32) return 'wheel';
  // Thin flat slabs => body panels / floor pans / tarps.
  if (minDim < maxDim * 0.18) return 'panel';
  // Small parts perched high on the vehicle are loose cargo.
  if (relY > 0.5 && spans < 0.3) return 'cargo';
  // Everything else: small odds-and-ends.
  return 'accessory';
}

// ── report ───────────────────────────────────────────────────────────────────
const ROLE_ORDER = ['frame', 'wheel', 'panel', 'cargo', 'accessory', 'ground'];
const ROLE_BLURB = {
  frame: 'skeleton — roll cage / chassis (strongest bonds; holds the shell together)',
  wheel: 'wheels / hubs (strong axle attachment to the frame)',
  panel: 'body panels, floor pans, seats (medium bonds)',
  cargo: 'strapped-on payload — barrels, crates, logs (weak; breaks off first)',
  accessory: 'small loose bits — chain, bucket (weakest; first to go)',
  ground: 'backdrop / ground plane from the source scene (excluded from the vehicle)',
};

function f(arr, w = 6) {
  return '[' + arr.map((x) => x.toFixed(2).padStart(w)).join(', ') + ']';
}

function main() {
  const { json, byteLength, jsonLen } = parseGlb(inputPath);
  const parts = extractParts(json);

  // whole-vehicle bounds, excluding any oversized ground plane
  const carParts = parts.filter((p) => Math.max(...p.size) < 50);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const p of carParts)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], p.wlo[k]);
      hi[k] = Math.max(hi[k], p.whi[k]);
    }
  const bounds = { lo, size: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]] };

  for (const p of parts) p.role = classifyPart(p, bounds);

  if (!quiet) {
    console.log(`\nGLB: ${inputPath}`);
    console.log(`  file ${(byteLength / 1e6).toFixed(2)} MB · JSON ${(jsonLen / 1e6).toFixed(2)} MB · ` +
      `nodes ${json.nodes?.length ?? 0} · meshes ${json.meshes?.length ?? 0} · materials ${json.materials?.length ?? 0}`);
    console.log(`  generator: ${json.asset?.generator ?? '?'}`);
    console.log(`\nVehicle bounds (ground excluded): size ${f(bounds.size)} m`);

    console.log(`\nParts (${parts.length}), grouped by role:`);
    for (const role of ROLE_ORDER) {
      const inRole = parts.filter((p) => p.role === role);
      if (!inRole.length) continue;
      console.log(`\n  ${role.toUpperCase()} — ${ROLE_BLURB[role]}`);
      inRole.sort((a, b) => a.center[0] - b.center[0]);
      for (const p of inRole) {
        console.log(
          `    ${p.name.padEnd(40)} tris=${String(p.tris).padStart(5)}  ` +
          `ctr=${f(p.center)} size=${f(p.size)}  mat=${p.material}`,
        );
      }
    }

    const counts = ROLE_ORDER.map((r) => `${r}:${parts.filter((p) => p.role === r).length}`).join('  ');
    console.log(`\nRole counts:  ${counts}`);
  }

  // ── manifest ────────────────────────────────────────────────────────────
  const manifest = {
    source: path.basename(inputPath),
    generatedBy: 'scripts/analyze-glb.mjs',
    bounds: { size: bounds.size, min: lo },
    roles: ROLE_ORDER.filter((r) => parts.some((p) => p.role === r)),
    parts: parts.map((p) => ({
      name: p.name,
      role: p.role,
      tris: p.tris,
      material: p.material,
      center: p.center.map((x) => +x.toFixed(4)),
      size: p.size.map((x) => +x.toFixed(4)),
    })),
  };

  const jsonOut = flags.get('json');
  if (jsonOut) {
    fs.writeFileSync(jsonOut, JSON.stringify(manifest, null, 2));
    if (!quiet) console.log(`\nWrote decomposition manifest -> ${jsonOut}`);
  }
  return manifest;
}

// Run as CLI when invoked directly.
const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (invokedDirectly) {
  try {
    main();
  } catch (err) {
    console.error('analyze-glb failed:', err.message);
    process.exit(1);
  }
}
