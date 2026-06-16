import RAPIER from '@dimforge/rapier3d-compat';
import fs from 'fs';
await RAPIER.init();
const asset = JSON.parse(fs.readFileSync(process.argv[2] || '/workspace/blast/js_stress_example/assets/buggy.pieces.json', 'utf8'));
const world = new RAPIER.World({ x: 0, y: 0, z: 0 }); // no gravity: only overlap pushes
const bodies = [];
for (const part of asset.parts) {
  for (const piece of part.pieces) {
    if (!piece.vertices?.length) continue;
    const n = piece.vertices.length;
    let cx=0,cy=0,cz=0; for (const v of piece.vertices){cx+=v[0];cy+=v[1];cz+=v[2];} cx/=n;cy/=n;cz/=n;
    const arr = new Float32Array(n*3);
    for (let i=0;i<n;i++){arr[i*3]=piece.vertices[i][0]-cx;arr[i*3+1]=piece.vertices[i][1]-cy;arr[i*3+2]=piece.vertices[i][2]-cz;}
    const desc = RAPIER.ColliderDesc.convexHull(arr);
    if (!desc) continue;
    const rb = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(cx, cy, cz));
    world.createCollider(desc, rb);
    bodies.push(rb);
  }
}
let peak = 0;
for (let s = 0; s < 120; s++) {
  world.step();
  let mx = 0;
  for (const b of bodies) { const v = b.linvel(); const sp = Math.hypot(v.x,v.y,v.z); if (sp>mx) mx=sp; }
  if (mx > peak) peak = mx;
}
console.log(`bodies=${bodies.length} peak speed (no gravity, overlap-only) = ${peak.toFixed(2)} m/s`);
console.log(peak < 1 ? 'OK: colliders do NOT overlap' : 'OVERLAP: colliders push apart (explosion source)');
