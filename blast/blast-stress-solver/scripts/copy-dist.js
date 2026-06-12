import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../js_stress_example/dist');
const dst = resolve(here, '../dist');

await mkdir(dst, { recursive: true });
await cp(resolve(src, 'stress_solver.wasm'), resolve(dst, 'stress_solver.wasm'));
await cp(resolve(src, 'stress_solver.cjs'), resolve(dst, 'stress_solver.cjs'));

// Copy the original ESM for Node/Electron usage if desired
await cp(resolve(src, 'stress_solver.mjs'), resolve(dst, 'stress_solver.mjs'));

// Create a browser-safe ESM by stubbing the dynamic import('module') used by Node.
// The exact statement emcc wraps it in changes between versions, so replace the bare
// import itself (gated behind ENVIRONMENT_IS_NODE, never executed in browsers/workers)
// rather than pattern-matching the whole block — a bare \`import("module")\` is harmless
// at runtime but fatal to bundlers consuming this file.
try {
  const esmSrc = await readFile(resolve(src, 'stress_solver.mjs'), 'utf8');
  const browserSafe = esmSrc.replaceAll(
    'await import("module")',
    '({/* node-only path stubbed for the browser build */})'
  );
  if (browserSafe.includes('import("module")')) {
    console.warn('[copy-dist] stress_solver.browser.mjs still references import("module") — strip pattern needs updating');
  }
  await writeFile(resolve(dst, 'stress_solver.browser.mjs'), browserSafe, 'utf8');
} catch (e) {
  // eslint-disable-next-line no-console
  console.warn('[copy-dist] Failed to create browser MJS variant:', e?.message || e);
}
