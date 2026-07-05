/**
 * Suite entry point shared by the shell script and the vitest suite. Runs a set of
 * city stress cases through the harness and returns the collected results. The
 * WASM-backed core factory is injected by the caller (`buildDestructibleCore` from
 * dist), keeping this module free of any runtime/WASM import so it can be bundled
 * for the Node script with esbuild (three.js externalized).
 */
import { CITY_SCENARIOS, CITY_SCENARIO_NAMES, type SizeTier } from './cityScenario';
import { runStressScenario, compareAB, type ABComparison, type BuildCore, type ScenarioResult } from './harness';

export interface SuiteOptions {
  buildCore: BuildCore;
  tier?: SizeTier;
  /** Subset of scenario keys to run (default: all). */
  only?: string[];
  /** Called after each scenario finishes (for progress logging). */
  onProgress?: (name: string, index: number, total: number) => void;
}

export async function runSuite(opts: SuiteOptions): Promise<ScenarioResult[]> {
  const { buildCore, tier = 'small', only, onProgress } = opts;
  const keys = (only && only.length ? only : CITY_SCENARIO_NAMES).filter((k) => CITY_SCENARIOS[k]);
  const results: ScenarioResult[] = [];

  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const useCase = CITY_SCENARIOS[key](tier);
    onProgress?.(useCase.name, i, keys.length);
    const result = await runStressScenario({
      name: useCase.name,
      scenario: useCase.scenario,
      buildCore,
      coreOpts: useCase.coreOpts,
      warmupFrames: useCase.warmupFrames,
      impactPlan: useCase.impactPlan,
      postImpactFrames: useCase.postImpactFrames,
    });
    results.push(result);
  }

  return results;
}

export interface IslandABOptions {
  buildCore: BuildCore;
  tier?: SizeTier;
  /** Which city preset to A/B (default: manyIslands — the obvious win candidate). */
  scenario?: string;
  /** Override the scenario's post-impact frame count. */
  postImpactFrames?: number;
  onProgress?: (label: string) => void;
}

/**
 * A/B the island-aware skip-settled solve against the default whole-graph solve on
 * the SAME scenario. Both arms track the per-frame bond trajectory so we can prove
 * the output is unchanged (or catch it if it isn't). Returns the comparison.
 */
export async function runIslandAB(opts: IslandABOptions): Promise<ABComparison> {
  const { buildCore, tier = 'small', scenario = 'manyIslands', postImpactFrames, onProgress } = opts;
  const factory = CITY_SCENARIOS[scenario];
  if (!factory) throw new Error(`unknown scenario '${scenario}' (have: ${CITY_SCENARIO_NAMES.join(', ')})`);

  // Build the scenario fresh per arm so neither shares mutable state.
  const baseCase = factory(tier);
  onProgress?.(`baseline (island solver OFF): ${baseCase.name}`);
  const baseline = await runStressScenario({
    name: `${baseCase.name} [whole-graph]`,
    scenario: baseCase.scenario,
    buildCore,
    coreOpts: baseCase.coreOpts,
    impactPlan: baseCase.impactPlan,
    postImpactFrames: postImpactFrames ?? baseCase.postImpactFrames,
    islandSolver: { enabled: false, skipSettled: false },
    trackBondTrajectory: true,
  });

  const treatCase = factory(tier);
  onProgress?.(`treatment (island solver ON + skipSettled): ${treatCase.name}`);
  const treatment = await runStressScenario({
    name: `${treatCase.name} [island+skip]`,
    scenario: treatCase.scenario,
    buildCore,
    coreOpts: treatCase.coreOpts,
    impactPlan: treatCase.impactPlan,
    postImpactFrames: postImpactFrames ?? treatCase.postImpactFrames,
    islandSolver: { enabled: true, skipSettled: true },
    trackBondTrajectory: true,
  });

  return compareAB(`island-aware solve on '${baseCase.name}' (${tier})`, baseline, treatment);
}

// ── Sleep / damping A/B ───────────────────────────────────────────────────────
// Named treatments for tuning how destructible chunks settle. Sleep thresholds use
// mode 'always' (safe: a higher threshold only bites near-zero velocity, so flight
// is untouched and a chunk resting on the 2nd story still benefits). Damping is
// gated to 'afterGroundCollision' (post-landing only) because damping applied in
// flight is what makes debris look floaty / stops it tumbling.
export const SLEEP_DAMPING_TREATMENTS: Record<string, Record<string, unknown>> = {
  sleepAlways: { sleepMode: 'always', sleepLinearThreshold: 0.5, sleepAngularThreshold: 0.5 },
  sleepGround: { sleepMode: 'afterGroundCollision', sleepLinearThreshold: 0.5, sleepAngularThreshold: 0.5 },
  sleepDampGround: {
    sleepMode: 'afterGroundCollision',
    sleepLinearThreshold: 0.5,
    sleepAngularThreshold: 0.5,
    smallBodyDamping: { mode: 'afterGroundCollision', minLinearDamping: 0.6, minAngularDamping: 0.6, colliderCountThreshold: 8 },
  },
  // Recommended combo: aggressive sleep everywhere (safe) + post-landing damping (floaty-safe).
  recommended: {
    sleepMode: 'always',
    sleepLinearThreshold: 0.5,
    sleepAngularThreshold: 0.5,
    smallBodyDamping: { mode: 'afterGroundCollision', minLinearDamping: 0.6, minAngularDamping: 0.6, colliderCountThreshold: 8 },
  },
};

export interface SleepDampingABOptions {
  buildCore: BuildCore;
  tier?: SizeTier;
  /** City preset to run (default: cascade — tall building, big debris that must settle). */
  scenario?: string;
  /** A treatment name from SLEEP_DAMPING_TREATMENTS, or a raw coreOpts override. */
  treatment?: string | Record<string, unknown>;
  /** Override the scenario's post-impact frame count (let the rubble fully settle). */
  postImpactFrames?: number;
  onProgress?: (label: string) => void;
}

/**
 * A/B engine-default settling vs tuned sleep-thresholds + post-landing damping on the
 * same scenario. Tracks both fracture parity (bonds) and MOTION parity (in-flight
 * speed/tumble) so we can prove debris flies the same while settling sooner.
 */
export async function runSleepDampingAB(opts: SleepDampingABOptions): Promise<ABComparison> {
  const { buildCore, tier = 'small', scenario = 'cascade', treatment = 'recommended', postImpactFrames, onProgress } = opts;
  const factory = CITY_SCENARIOS[scenario];
  if (!factory) throw new Error(`unknown scenario '${scenario}' (have: ${CITY_SCENARIO_NAMES.join(', ')})`);
  const treatName = typeof treatment === 'string' ? treatment : 'custom';
  const treatOpts = typeof treatment === 'string' ? SLEEP_DAMPING_TREATMENTS[treatment] : treatment;
  if (!treatOpts) throw new Error(`unknown treatment '${treatment}' (have: ${Object.keys(SLEEP_DAMPING_TREATMENTS).join(', ')})`);

  const baseCase = factory(tier);
  onProgress?.(`baseline (engine defaults): ${baseCase.name}`);
  const baseline = await runStressScenario({
    name: `${baseCase.name} [defaults]`,
    scenario: baseCase.scenario,
    buildCore,
    coreOpts: baseCase.coreOpts,
    impactPlan: baseCase.impactPlan,
    postImpactFrames: postImpactFrames ?? baseCase.postImpactFrames,
    trackBondTrajectory: true,
    trackKinematics: true,
  });

  const treatCase = factory(tier);
  onProgress?.(`treatment (${treatName}): ${treatCase.name}`);
  const treated = await runStressScenario({
    name: `${treatCase.name} [${treatName}]`,
    scenario: treatCase.scenario,
    buildCore,
    coreOpts: { ...baseCase.coreOpts, ...treatOpts },
    impactPlan: treatCase.impactPlan,
    postImpactFrames: postImpactFrames ?? treatCase.postImpactFrames,
    trackBondTrajectory: true,
    trackKinematics: true,
  });

  return compareAB(`sleep/damping on '${baseCase.name}' (${tier}, ${treatName})`, baseline, treated, [
    'rapierStepMs',
    'solverSolveMs',
    'contactInjectSplashMs',
    'snapshotCaptureMs',
  ]);
}

export { CITY_SCENARIO_NAMES } from './cityScenario';
export { printReport, toJsonReport, printABReport, abToJsonReport } from './harness';
