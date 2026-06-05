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
  onProgress?: (label: string) => void;
}

/**
 * A/B the island-aware skip-settled solve against the default whole-graph solve on
 * the SAME scenario. Both arms track the per-frame bond trajectory so we can prove
 * the output is unchanged (or catch it if it isn't). Returns the comparison.
 */
export async function runIslandAB(opts: IslandABOptions): Promise<ABComparison> {
  const { buildCore, tier = 'small', scenario = 'manyIslands', onProgress } = opts;
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
    postImpactFrames: baseCase.postImpactFrames,
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
    postImpactFrames: treatCase.postImpactFrames,
    islandSolver: { enabled: true, skipSettled: true },
    trackBondTrajectory: true,
  });

  return compareAB(`island-aware solve on '${baseCase.name}' (${tier})`, baseline, treatment);
}

export { CITY_SCENARIO_NAMES } from './cityScenario';
export { printReport, toJsonReport, printABReport, abToJsonReport } from './harness';
