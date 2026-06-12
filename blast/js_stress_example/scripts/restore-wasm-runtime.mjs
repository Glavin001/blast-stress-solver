// Restore the Emscripten runtime artifacts in dist/ after `tsc` (build:ts) runs.
//
// The demo tsconfig compiles a legacy root module (`stress.js`) that imports
// `./stress_solver.cjs`, so tsc pulls the Emscripten JS glue into its program and
// re-emits a DOWNLEVELED copy to `dist/stress_solver.cjs` — silently replacing the
// real glue next to the (untouched) `dist/stress_solver.wasm`. The transformed glue
// is not equivalent; anything that subsequently copies dist (e.g. blast-stress-solver's
// copy-dist.js) then ships a corrupted glue/wasm pair that loads but mis-marshals.
//
// `scripts/build.js` maintains pristine copies of all three artifacts at the project
// root on every emcc build, so restoring dist from those after tsc keeps dist truthful.
import { copyFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const distDir = resolve(projectRoot, 'dist');

let restored = 0;
for (const name of ['stress_solver.cjs', 'stress_solver.mjs', 'stress_solver.wasm']) {
  const src = resolve(projectRoot, name);
  const dst = resolve(distDir, name);
  if (!existsSync(src)) continue;
  copyFileSync(src, dst);
  restored += 1;
}

if (restored > 0) {
  console.log(`[restore-wasm-runtime] restored ${restored} Emscripten artifact(s) in dist/ after tsc`);
} else {
  console.warn('[restore-wasm-runtime] no root stress_solver.* artifacts found — run scripts/build.js (emcc) first');
}
