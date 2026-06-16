# AGENTS.md

## Cursor Cloud specific instructions

In Cursor Cloud VMs the repository is checked out at **`/workspace`** (substitute for `/home/user/PhysX` in the commands below).

**Primary branch:** `feat/rapier-destruction` (CI push + production demos). Check out and pull this branch before setup/build; do not use `main` for Blast stress-solver work unless explicitly asked.

### Project overview

This is the NVIDIA PhysX monorepo containing PhysX, Blast, and Flow SDKs. The active development focus is on:

- the **Blast Stress Solver JavaScript/WASM library** (`blast/blast-stress-solver`)
- the **Blast Stress Solver Rust crate** (`blast/blast-stress-solver-rs`)
- Three.js + Rapier browser demos (`blast/js_stress_example`)
- the packaged-consumer Bevy/Rapier demo (`blast/blast-stress-demo-rs`)

### Prerequisites

- **Node.js** (v22+) is available via nvm.
- **Emscripten SDK** is required for WASM compilation. It should be installed at `/opt/emsdk`. If not present, install it:
  ```bash
  git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk
  cd /opt/emsdk
  ./emsdk install 3.1.51
  ./emsdk activate 3.1.51
  source /opt/emsdk/emsdk_env.sh
  ```
  The `~/.bashrc` should source it automatically. Verify with `emcc --version`.
- If `/opt/emsdk` was cloned with `sudo`, fix ownership before `./emsdk install`: `sudo chown -R "$(whoami)" /opt/emsdk`.

### First-time setup

```bash
# 1. Install everything needed for the demos
cd /home/user/PhysX
npm run setup:demos
```

### Fast path for demos

```bash
# Ensure emsdk is sourced
source /opt/emsdk/emsdk_env.sh

# Build everything required for the browser demos and start the server
cd /home/user/PhysX
npm start   # checks toolchain/deps, builds assets, then serves http://localhost:8000
```

If the assets are already built and you only need the server:

```bash
cd /home/user/PhysX
npm run serve:demos
```

### Full build (from scratch)

```bash
# Ensure emsdk is sourced
source /opt/emsdk/emsdk_env.sh

# Build the blast-stress-solver package (WASM + TypeScript)
cd /home/user/PhysX/blast/blast-stress-solver
npm run build

# Build browser demo assets (bridge TS + esbuild entrypoints)
cd ../js_stress_example
npm run build:web

# Run tests
cd ../blast-stress-solver
npm test

# Serve demos without rebuilding
cd /home/user/PhysX
npm run serve:demos   # http://localhost:8000
```

### Rust Crate Release Flow

The Rust crate release flow is staged-package based. Do **not** run:

```bash
cargo publish --manifest-path blast/blast-stress-solver-rs/Cargo.toml
```

The source crate at `blast/blast-stress-solver-rs` is marked `publish = false`
on purpose. Local and CI publishes must go through the staged package scripts.

Local crates.io preflight:

```bash
cd /Users/glavin/Development/PhysX
scripts/publish-blast-stress-solver.sh --dry-run
```

Local publish:

```bash
cd /Users/glavin/Development/PhysX
scripts/publish-blast-stress-solver.sh
```

Tag-driven GitHub Actions release:

1. Bump `blast/blast-stress-solver-rs/Cargo.toml` to the new version.
2. Commit and push the branch.
3. Push a matching tag:

```bash
git tag blast-stress-solver-v<version>
git push origin blast-stress-solver-v<version>
```

The release workflow at `.github/workflows/release-blast-stress-solver.yml`
stages the crate, runs the packaged native/wasm/demo-consumer proofs, runs
`cargo publish --dry-run`, publishes to crates.io, and creates a GitHub
release. It requires the repository secret `CARGO_REGISTRY_TOKEN`.

### Build chain details

1. `cd blast/js_stress_example && npm run build` — compiles C++ stress solver to WASM via `emcc`, outputs `dist/stress_solver.{cjs,mjs,wasm}`. Takes ~20 seconds (two targets: node-cjs + browser-esm).
2. `cd blast/blast-stress-solver && npm run build` — runs step 1 as `prebuild`, then bundles TypeScript with `tsup`, copies WASM to `dist/`
3. `cd blast/js_stress_example && npm run build:web` — runs `build:ts` for the bridge-style TS outputs and `build:demo:*` for `wall-demolition`, `tower-collapse`, and `fractured-wall`. The `build:ts` step tolerates the pre-existing TypeScript errors because `tsc` still emits usable JS, and those three entrypoints use import-map bare specifiers so they are rebuilt with **esbuild** instead of relying on the emitted `tsc` output.
4. `cd /home/user/PhysX && npm run build:demos` — runs the blast-stress-solver build plus `blast/js_stress_example`'s `build:web` script from the repo root.
5. `cd /home/user/PhysX && npm run check:demos` — verifies that native optional dependencies (`rollup`, `esbuild`) match the current platform and that Emscripten is available when the WASM runtime needs to be rebuilt.
6. `npm start` (root) — runs `npm run check:demos`, then `npm run build:demos`, then starts the static file server on port 8000. Use `npm run serve:demos` if you only want to serve already-built assets.

### Running tests

- `cd blast/blast-stress-solver && npm test` — runs all vitest tests (authoring, gravity, damage, headless scenarios, integration, Three.js adapter)
- Key test files in `blast/blast-stress-solver/src/tests/`:
  - `rapier.headless-scenarios.test.ts` — gravity stability, projectile collisions, material strength, catastrophic vs partial damage, damage toggle, parameter sweeps, bond inspection, structure-specific behavior, determinism, API surface, scenario builder correctness (including bond isotropy verification)
  - `rapier.integration.test.ts` — full WASM+Rapier pipeline (fracture, projectiles, damage, profiler)
  - `rapier.damage.test.ts` — damage system unit tests
  - `authoring.*.test.ts` — bond authoring / triangle bonding tests
- `cd blast/js_stress_example && npm run test:split` — split/fracture integration tests (vitest)
- `cd blast/js_stress_example && npm run test:bridge-ui` — Playwright browser tests (requires `npx playwright install chromium` first)

### Browser demos

After running `npm start` at the root:

**Primary demos (high-level API):**
- **`/blast/js_stress_example/destructible-vehicle.html`** — Loads a GLB model (`assets/buggy.glb`), auto-decomposes it into parts, classifies each into a structural role (frame/wheel/panel/cargo/accessory) and bonds them in a **strength hierarchy** so the payload sheds first while the cage holds; a hard hit/drop shatters it all. Decomposition lives in `glb-vehicle.ts`; the companion CLI `scripts/analyze-glb.mjs` reports/classifies any GLB and emits a hand-editable manifest. See `DESTRUCTIBLE_VEHICLE.md`.
- **`/blast/js_stress_example/wall-demolition.html`** — Destructible brick wall. Click to shoot projectiles. Config panel for wall geometry, projectile params, material scale (log slider), and **Auto Bonds (experimental)** toggle.
- **`/blast/js_stress_example/tower-collapse.html`** — Destructible tower. Same config panel pattern. Includes diagonal bonds and small-body damping.
- **`/blast/js_stress_example/fractured-wall.html`** — Voronoi-fractured wall using `three-pinata`. Irregular fragments with proximity-based bond detection. Config panel for fragment count, projectile params, and material scale.

**Bridge demos:**
- **`bridge-split-demo.html`** — Fully featured destructible bridge with real-time config, projectile spawning, and full fracture into independent physics bodies. Included in `npm run build:web`.
- `bridge-ext.html` — Older two-phase fracture demo (also included in `npm run build:web`)

**Legacy:** `bridge-demo.html` — Stress coloring only; fracture disabled at `simulation.js:117`.

### Architecture notes

**Stress solver paths for bond breaking:**
1. **Gravity stress** — `solver.addActorGravity()` + `solver.update()` computes bond stress from gravity. Overstressed bonds are fractured via `generateFractureCommandsPerActor()` + `applyFractureCommands()`.
2. **Contact force injection** — Rapier contact forces from projectile impacts are rotated to body-local space and injected via `solver.addForce()` with splash radius, so the stress solver sees collision impacts too. Controlled by `contactForceScale` option in `destructible-core.ts`.
3. **Damage system** — Contact forces -> `damageSystem.onImpact()` -> node health decrement -> bond removal. Separate from the stress solver. Disabled by default in wall/tower demos.

**Bond normalization:** Area normalization in scenario builders (`wallScenario.ts`, `towerScenario.ts`, `bridgeScenario.ts`) uses **isotropic scaling** — a single uniform scale factor (geometric mean of per-axis scales) to avoid directional bias. Per-axis normalization causes horizontal layer separation under gravity because it makes vertical bonds stronger than horizontal bonds.

**Auto-bonding (experimental):** `applyAutoBondingToScenario()` from `blast-stress-solver/three` replaces manual grid bonds with geometry-derived bonds computed by the WASM solver's `createBondsFromTriangles` API. Uses actual triangle-mesh shared surfaces for bond area/normal calculation. Requires `scenario.parameters.fragmentGeometries` (array of `THREE.BufferGeometry` per node). Toggle available in wall/tower demo UIs.

**Key source locations:**
- Scenario builders: `blast/blast-stress-solver/src/scenarios/` (wallScenario.ts, towerScenario.ts, bridgeScenario.ts)
- Core fracture logic: `blast/blast-stress-solver/src/rapier/destructible-core.ts` (`processOneFracturePass`, contact force injection, bond tracking)
- WASM bridge: `blast/blast-stress-solver/src/stress.ts` (addForce, addActorGravity, generateFractureCommandsPerActor, applyFractureCommands, createBondsFromTriangles)
- Auto-bonding: `blast/blast-stress-solver/src/three/autoBonding.ts`
- Damage system: `blast/blast-stress-solver/src/rapier/damage.ts`

### Gotchas

- The `blast/blast-stress-solver` build has benign `import.meta` CJS warnings from tsup — these do not affect functionality.
- TypeScript strict checking (`tsc --noEmit`) in `blast/js_stress_example` shows pre-existing type errors; `noEmitOnError: false` in tsconfig ensures files are still emitted, and `npm run build:ts` intentionally continues so `npm run build:web` can finish.
- The WASM build takes ~20 seconds per run (two targets: node-cjs + browser-esm).
- `npm run setup:demos` installs root dependencies, `blast-stress-solver` dependencies with `--ignore-scripts`, and `js_stress_example` dependencies in one step.
- `npm install --ignore-scripts` is used for `blast/blast-stress-solver` during dependency refresh to avoid triggering a full rebuild on install.
- `blast/js_stress_example`'s `build:web` script uses **esbuild** for `wall-demolition`, `tower-collapse`, and `fractured-wall` because those demos rely on bare import specifiers resolved by the HTML import map at runtime.
- If `npm start` fails in `check:demos` with a native dependency mismatch, reinstall the affected package's `node_modules` on the current machine instead of copying `node_modules` across platforms.
- Projectile TTL is in **wall-clock seconds** (via `performance.now()`), not simulation time. In headless tests, use very short TTL values (e.g., 0.001) since physics steps execute much faster than real time.
- `getActiveBondsCount()` uses JS-side tracking (`bondTable.length - removedBondIndices.size`) which may lag behind the WASM solver's internal bond state after `applyFractureCommands`. Multiple resimulation passes (controlled by `maxResimulationPasses`) help propagate fractures.
- Bond areas directly affect stress: `stress = force / area`. Larger area = lower stress = harder to break. This is why isotropic normalization matters — asymmetric areas create directional weakness.
- **Missing `stress_solver.cjs` / `stress_solver.wasm`:** Tests that load the WASM stress solver (authoring, gravity, damage, integration) will fail with `Cannot find module './stress_solver.cjs'` if the C++ → WASM build has not been run. This happens when the Emscripten SDK (`emcc`) is not installed or not on `PATH`. To resolve:
  1. Install and activate emsdk (see **Prerequisites** above).
  2. Run `cd blast/blast-stress-solver && npm run build` (the `prebuild` step compiles the WASM).
  3. Alternatively, set `BLAST_STRESS_SOLVER_SKIP_WASM_BUILD=1` to skip the WASM compilation and iterate on TypeScript only — but WASM-dependent tests will still fail.
  
  The 7 test files that do **not** require the WASM binary (scenario builders, Three.js adapter, bundle exports, split migrator, headless scenarios) will pass regardless.

### CI

- GitHub Actions workflow at `.github/workflows/ci.yml`
- **Push** trigger: only runs on `feat/rapier-destruction` branch (production)
- **Pull request** trigger: runs on all PRs
- This prevents double CI runs when PRs are opened from feature branches

## Destructible-scene tuning — hard rules (stability must be REAL, not faked)

These are non-negotiable when building destructible scenes (e.g. the brick-castle
demo) on the Blast stress solver. Stability must come from the scene being
physically sound, NOT from masking instability.

- **NEVER use an `onImpact` handler to drive breaking.** (The `onImpact` path is
  being removed.) Destruction must emerge from the simulation, not from bespoke
  hand-cut bonds on contact.
- **NEVER raise `materialScale` to gain stability.** A high `materialScale` that
  makes a structure "stand" is an unacceptable hack/cheat — it just makes
  everything too strong to break, and hides that the structure isn't actually
  self-supporting. Tune real, physical quantities (e.g. per-bond contact `area`,
  geometry/contact) instead.
- **The structure must be self-supporting as plain rigid bodies.** Before relying
  on the stress solver, verify the scene SETTLES when every (non-support) chunk is
  an independent dynamic rigid body with NO bonds — just gravity + friction +
  contact. If it collapses with no bonds, the geometry is broken (gaps, no
  bedding, too thin / too tall, no interlock) — fix the geometry. The stress solver
  is then a performance OPTIMIZATION over an already-stable rigid-body pile (group
  settled regions as one body), never a crutch that holds up an unstable one.
- Island-skipping / lazy colliders are performance optimizations only. They must
  not be used to keep an unstable structure from being computed (that just defers
  the collapse to the first disturbance). They are legitimate only when the region
  is genuinely stable.

## Auto-bonding (`createBondsFromTriangles`) — gotchas that cost hours

The triangle auto-bonder has two modes with very different behaviour. Verified
empirically by feeding it two boxes separated along each axis (see the brick
castle work):

- **`'average'` mode silently drops EVERY vertical (Y-normal) contact.** It bonds
  side-by-side (X/Z-normal) faces fine but never bonds a box resting on top of
  another — even at gap 0. Symptom: a stacked/coursed structure (wall, tower,
  castle) auto-bonds with plenty of horizontal bonds but almost no course-to-course
  bonds, so the stress solver sees disconnected courses and the whole thing
  collapses under self-weight on the first frame. This is NOT a stability/tuning
  problem; it is missing bonds. Always check the vertical-vs-horizontal bond split
  (bond normal · up) — if vertical ≈ 0 on a coursed structure, this is why.
- **`'exact'` mode bonds all axes** (including vertical) **but only when the
  touching faces are coincident (separation ~0).** A few-mm gap → 0 bonds. It does
  real coplanar-overlap area (running-bond half-offset → half-area bond), so it is
  the right choice for axis-aligned masonry: place bricks TOUCHING (mortarGap 0) so
  adjacent faces coincide, then use `mode:'exact'`. This also bonds Voronoi
  intra-brick faces (already coincident) for free.
- `maxSeparation` only applies to `'average'`. It is ignored by `'exact'`.
- Quick probe to characterise the bonder before trusting it: two unit boxes,
  separate by `size+gap` along x, then y, then z, and print `bonds`/`area`/`normal`
  for each mode. Cheap, deterministic, and exposes the Y-axis hole immediately.

## Headless verification recipes (brick-castle)

- **Bond/overlap/support audit** (no browser): import the built `dist/scenarios.js`
  + `dist/three.js`, `buildBrickCastleScenario({...})`, then on the returned
  `scenario` check: (a) inter-brick collider penetration via a spatial hash over
  `nodes[i].centroid` ± `fragmentSizes[i]/2` (skip same-`brickIdByNode` siblings) —
  must be 0 past ~0.1 m; (b) per-brick bond degree from `scenario.bonds` mapped
  through `brickIdByNode` — every non-support brick must have ≥1 inter-brick bond;
  (c) `tierCounts`. Runs in ~1 s and catches the vertical-bond hole and authored
  overlaps without spinning up physics.
- **Stability/siege soak** (`scripts/soak-castle.mjs`, Playwright): the settle phase
  is the key gate — `peakSpeed ≈ 0` and `broken ≈ 0/total` while UNTOUCHED proves
  the structure stands on bonds alone. Then siege phases assert collapse without
  explosion (no body > ~120 m/s, 0 bodies > 60 m/s). Serve the built
  `.vercel/output/static` (flat paths, e.g. `/brick-castle.html`).
- **Measuring "did it move"**: read `core.chunks[i].worldPosition` (dormant chunks
  keep their initial position). Do NOT read `world.getRigidBody(handle).translation()`
  for chunks parented to a static root — it returns the root's transform (bogus
  large displacement). And `brokenBonds` alone is a misleading "it's fixed" signal.
- Building a scenario CONSUMES `fragmentGeometries`. Build a FRESH scenario per
  core/test; reusing one across cores gives false negatives.
