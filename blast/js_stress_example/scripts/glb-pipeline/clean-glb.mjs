/**
 * clean-glb.mjs — glTF-Transform weld + simplify pre-pass (the gltfjsx --transform
 * idea) before the CoACD pipeline.
 *
 * The source buggy is ~96% non-watertight triangle soup, which makes CoACD produce
 * thin sliver pieces that can't be de-interpenetrated. Welding merges the coincident
 * verts into manifold-ish parts; meshopt simplification removes the thin features and
 * cuts poly count (faster CoACD, chunkier clippable pieces). dedup + prune tidy up.
 *
 *   node clean-glb.mjs in.glb out.glb [--ratio 0.3] [--error 0.01] [--weld 0.0005]
 */
import { NodeIO } from '@gltf-transform/core';
import { dedup, weld, simplify, prune } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';

const args = process.argv.slice(2);
const input = args.find((a) => a.endsWith('.glb') && !args[args.indexOf(a) - 1]?.startsWith('--')) || 'assets/buggy.glb';
const output = args.filter((a) => a.endsWith('.glb'))[1] || input.replace(/\.glb$/, '-clean.glb');
const ratio = Number(args[args.indexOf('--ratio') + 1] ?? 0.3);
const error = Number(args[args.indexOf('--error') + 1] ?? 0.01);
const weldTol = Number(args[args.indexOf('--weld') + 1] ?? 0.0005);

await MeshoptSimplifier.ready;
const io = new NodeIO();
const doc = await io.read(input);

const countTris = () => doc.getRoot().listMeshes()
  .flatMap((m) => m.listPrimitives())
  .reduce((n, p) => n + (p.getIndices()?.getCount() ?? p.getAttribute('POSITION')?.getCount() ?? 0) / 3, 0);
const before = countTris();

await doc.transform(
  dedup(),
  weld({ tolerance: weldTol }),
  simplify({ simplifier: MeshoptSimplifier, ratio, error, lockBorder: false }),
  prune(),
);

const after = countTris();
await io.write(output, doc);
console.log(`[clean-glb] ${input} -> ${output}`);
console.log(`[clean-glb] weld=${weldTol} simplify ratio=${ratio} error=${error}`);
console.log(`[clean-glb] triangles ${Math.round(before).toLocaleString()} -> ${Math.round(after).toLocaleString()} (${(100 * after / before).toFixed(0)}%)`);
