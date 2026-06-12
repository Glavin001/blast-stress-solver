/**
 * **Mini-city bottleneck regression** — runs the recording analyzer on a real,
 * committed session capture (`fixtures/mini-city.profile.sim.json.gz`, a
 * profile-only strip of a 709-frame / ~1300-body mini-city demolition) and locks
 * in the *shape* of where real-time runtime goes.
 *
 * This is the test that answers "what are our bottlenecks on a large, complex
 * scene?" with numbers from the actual scene, and guards them against drift:
 *
 *   • rapier `world.step()` is the single dominant phase (debris collision).
 *   • the stress solve is #2, and inside it the WASM CGNR solve dominates.
 *   • resimulation (rollback + re-step on fracture) runs on a large share of
 *     frames and roughly doubles them — the systemic "second physics pass" lever.
 *   • frame cost scales with rigid-body (debris) count — rapierStep most of all.
 *   • the `contactInjectGrid` per-frame rebuild stays ~0 (a previously-landed
 *     optimisation we don't want to silently regress).
 *
 * The assertions use wide tolerances: they assert the *bottleneck ranking and
 * order of magnitude*, not exact millisecond values (which are machine- and
 * build-dependent). They fail only if the performance *character* of the scene
 * changes — which is exactly when a perf-minded reviewer wants to look.
 *
 * The fixture is profile-only (per-frame body trajectory stripped) so it stays a
 * ~56 KB artifact; regenerate with `node scripts/strip-recording.mjs`.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeSimRecording, type SimRecordingExport } from '../rapier/sessionRecorder';
import { analyzeRecording, formatAnalysisReport } from '../rapier/recordingAnalysis';

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, 'fixtures/mini-city.profile.sim.json.gz');

function loadFixture(): SimRecordingExport {
  const raw = gunzipSync(readFileSync(fixturePath));
  return JSON.parse(raw.toString('utf8')) as SimRecordingExport;
}

describe('mini-city recording — bottleneck regression', () => {
  const bundle = loadFixture();
  const decoded = decodeSimRecording(bundle);
  const analysis = analyzeRecording(decoded, {
    meta: bundle.meta as Record<string, unknown> | undefined,
  });

  it('prints the full bottleneck report (visible with --reporter=verbose)', () => {
    // Surfaced for humans reading CI logs / running locally; not an assertion.
    console.log('\n' + formatAnalysisReport(analysis));
    expect(analysis.frames).toBe(709);
  });

  it('is the expected large, complex mini-city scene', () => {
    expect(analysis.demo).toBe('mini-city');
    // Debris accumulates from a near-empty scene to >1000 rigid bodies.
    expect(analysis.scene.rigidBodies.max).toBeGreaterThan(1000);
    expect(analysis.scene.activeBonds.max).toBeGreaterThan(10000);
    // It is a heavy scene: the median frame misses the 60 FPS budget at some point.
    expect(analysis.budget.pctMiss60).toBeGreaterThan(10);
  });

  it('rapier world.step() is the single dominant phase', () => {
    const leaves = analysis.phases.filter((p) => p.kind === 'leaf');
    expect(leaves[0].field).toBe('rapierStepMs');
    // It alone is a large minority of every frame…
    expect(leaves[0].shareOfTotalPct).toBeGreaterThan(30);
    // …and bigger than the #2 leaf (the stress solve).
    expect(leaves[0].stats.mean).toBeGreaterThan(leaves[1].stats.mean);
    expect(leaves[1].field).toBe('solverUpdateMs');
  });

  it('inside the stress solve, the WASM CGNR solve dominates over JS injection', () => {
    const find = (f: string) => analysis.phases.find((p) => p.field === f)?.stats.mean ?? 0;
    const cgnr = find('solverSolveMs');
    const contactInject = find('solverContactInjectMs');
    const gravityInject = find('solverGravityInjectMs');
    expect(cgnr).toBeGreaterThan(contactInject);
    expect(cgnr).toBeGreaterThan(gravityInject);
    // CGNR is the majority of the solver's leaf total.
    expect(cgnr).toBeGreaterThan(find('solverUpdateMs') * 0.5);
  });

  it('resimulation is a systemic lever: fires often and ~doubles those frames', () => {
    const r = analysis.resim;
    // A large share of frames run a second (rollback + re-step) pass.
    expect(r.resimFraction).toBeGreaterThan(0.25);
    // The second pass is a material fraction of all wall-clock time.
    expect(r.resimShareOfWallClockPct).toBeGreaterThan(15);
    // A resim frame costs noticeably more than a body-count-matched non-resim
    // frame — close to 2× (the whole physics+solve runs again).
    expect(r.matchedResimMultiplier).toBeGreaterThan(1.5);
    expect(r.matchedResimMultiplier).toBeLessThan(2.6);
  });

  it('frame cost scales with rigid-body (debris) count — rapierStep most of all', () => {
    const totalScale = analysis.scaling.find((s) => s.field === 'totalMs')!;
    const rapierScale = analysis.scaling.find((s) => s.field === 'rapierStepMs')!;
    // Positive, well-correlated growth with body count.
    expect(totalScale.msPer100Bodies).toBeGreaterThan(0);
    expect(rapierScale.msPer100Bodies).toBeGreaterThan(0);
    expect(rapierScale.fit.r2).toBeGreaterThan(0.5);
    // rapierStep is the biggest single contributor to that growth.
    const solverScale = analysis.scaling.find((s) => s.field === 'solverUpdateMs')!;
    expect(rapierScale.msPer100Bodies).toBeGreaterThan(solverScale.msPer100Bodies);
  });

  it('the per-frame contact-inject grid rebuild stays eliminated (~0)', () => {
    // A previously-landed optimisation: the splash grid is no longer rebuilt every
    // frame. Guard it so a regression that reintroduces per-frame grid work shows up.
    const grid = analysis.phases.find((p) => p.field === 'contactInjectGridMs');
    // Either pruned as a dead phase (mean<1e-4) or present but negligible.
    expect(grid?.stats.mean ?? 0).toBeLessThan(0.05);
  });

  it('the worst frame is the one-time startup fracture-cascade hitch', () => {
    // The single biggest spike is the first big collider rebuild, not steady state.
    const colliderHitch = analysis.hitches.find((h) => h.field === 'colliderRebuildMs');
    expect(colliderHitch).toBeDefined();
    expect(colliderHitch!.ms).toBeGreaterThan(50);
    // It is an early-session event, not a recurring steady-state cost.
    expect(colliderHitch!.frameIndex).toBeLessThan(150);
  });
});
