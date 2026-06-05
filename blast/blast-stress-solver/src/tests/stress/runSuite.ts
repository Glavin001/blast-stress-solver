/**
 * Suite entry point shared by the shell script and the vitest suite. Runs a set of
 * city stress cases through the harness and returns the collected results. The
 * WASM-backed core factory is injected by the caller (`buildDestructibleCore` from
 * dist), keeping this module free of any runtime/WASM import so it can be bundled
 * for the Node script with esbuild (three.js externalized).
 */
import { CITY_SCENARIOS, CITY_SCENARIO_NAMES, type SizeTier } from './cityScenario';
import { runStressScenario, type BuildCore, type ScenarioResult } from './harness';

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

export { CITY_SCENARIO_NAMES } from './cityScenario';
export { printReport, toJsonReport } from './harness';
