// Compare the two fracturers on one wall-sized box, on the four things that
// decide whether a switch is worth it: seam width, convexity, watertightness
// and how many distinct shapes come out.
//
// Bond areas are measured by NvBlast's own auto-bonder in EXACT mode -- the
// same tool used to audit the current backend's bonds -- so neither fracturer
// is scored by its own arithmetic.
import * as THREE from 'three';
import { generateAutoBondsFromChunks } from './dist/three.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';
import { voronoiCells, polygonCentroid, polygonArea } from './scripts/export-fractured-city.mjs';

const W = 4.0, H = 2.5, T = 0.8;   // a wall segment
const N = 12;                       // shards

function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

// ── our 2D backend, as the builder does it: cells in the cross-section, extruded
function ours(seed) {
  const rng = mulberry32(seed);
  const gu = Math.max(1, Math.round(Math.sqrt(N * (W / H))));
  const gw = Math.max(1, Math.ceil(N / gu));
  let seeds = [];
  for (let a = 0; a < gu && seeds.length < N; a++)
    for (let b = 0; b < gw && seeds.length < N; b++)
      seeds.push([(a + 0.25 + rng() * 0.5) / gu * W - W/2, (b + 0.25 + rng() * 0.5) / gw * H - H/2]);
  for (let pass = 0; pass < 2; pass++) {           // Lloyd, as shipped
    const cs = voronoiCells(seeds, -W/2, -H/2, W/2, H/2);
    seeds = seeds.map((s, i) => (cs[i] && cs[i].length >= 3 ? polygonCentroid(cs[i]) : s));
  }
  const cells = voronoiCells(seeds, -W/2, -H/2, W/2, H/2);
  return cells.filter(c => c.length >= 3 && polygonArea(c) > 1e-4).map(poly => {
    const pts = [];
    for (const [u, v] of poly) { pts.push(new THREE.Vector3(u, v, -T/2), new THREE.Vector3(u, v, T/2)); }
    return { geometry: new ConvexGeometry(pts), matrix: new THREE.Matrix4() };
  });
}

// ── three-pinata, 3D voronoi, seeded
async function pinata(seed) {
  const { DestructibleMesh, FractureOptions } = await import('@dgreenheck/three-pinata');
  const mesh = new DestructibleMesh(new THREE.BoxGeometry(W, H, T), new THREE.MeshBasicMaterial());
  // Fragments come back RECENTRED, with the offset on the mesh -- so they must
  // be placed by their matrix or every one of them sits at the origin and the
  // bonder finds no contacts at all.
  return mesh.fracture(new FractureOptions({
    fractureMethod: 'voronoi', fragmentCount: N, seed,
  })).map(f => { f.updateMatrix(); return { geometry: f.geometry, matrix: f.matrix.clone() }; });
}

function hullOf(geo) {
  const p = geo.getAttribute('position');
  const pts = [];
  for (let i = 0; i < p.count; i++) pts.push([p.getX(i), p.getY(i), p.getZ(i)]);
  return pts;
}

// Volume of a closed triangle mesh via the divergence theorem. A watertight
// piece has a well-defined, positive volume; an open one does not.
function meshVolume(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  const p = g.getAttribute('position');
  let v = 0;
  for (let i = 0; i < p.count; i += 3) {
    const ax=p.getX(i),ay=p.getY(i),az=p.getZ(i);
    const bx=p.getX(i+1),by=p.getY(i+1),bz=p.getZ(i+1);
    const cx=p.getX(i+2),cy=p.getY(i+2),cz=p.getZ(i+2);
    v += (ax*(by*cz-bz*cy) - ay*(bx*cz-bz*cx) + az*(bx*cy-by*cx)) / 6;
  }
  return Math.abs(v);
}

async function score(label, parts) {
  const chunks = parts.map(p => ({ geometry: p.geometry, matrix: p.matrix }));
  const geos = parts.map(p => p.geometry);
  let bonds = [];
  try {
    bonds = await generateAutoBondsFromChunks(chunks, { mode: 'EXACT' }) ?? [];
  } catch (e) { console.log('  autobond failed:', String(e).slice(0, 160)); }
  const areas = bonds.map(b => b.area).sort((a, b) => a - b);
  const widths = areas.map(a => a / H);          // seam width if it spans the height
  const tiny = widths.filter(w => w < 0.05).length;
  const vols = geos.map(meshVolume);
  const totalVol = vols.reduce((a, b) => a + b, 0);
  const boxVol = W * H * T;
  const verts = geos.map(g => hullOf(g).length);
  console.log(`\n${label}`);
  console.log(`  fragments        ${geos.length}`);
  console.log(`  bonds found      ${bonds.length}`);
  console.log(`  seam width p10   ${areas.length ? (widths[Math.floor(widths.length/10)]*100).toFixed(1) : '-'} cm`);
  console.log(`  seam width median${areas.length ? (widths[Math.floor(widths.length/2)]*100).toFixed(1) : '-'} cm`);
  console.log(`  seams under 5 cm ${areas.length ? (100*tiny/widths.length).toFixed(0) : '-'}%`);
  console.log(`  volume recovered ${(100*totalVol/boxVol).toFixed(1)}%  (100 = tiles the box exactly)`);
  console.log(`  verts per piece  min ${Math.min(...verts)}, max ${Math.max(...verts)}`);
}

await score('ours: 2D voronoi + extrude (shipping)', ours(1234));
await score('three-pinata: 3D voronoi (seeded)', await pinata(1234));
