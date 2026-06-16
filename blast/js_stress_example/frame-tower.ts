/**
 * "Multi-storey frame tower" destructible demo (vibe-city /destructible-stress preset).
 * Thin entrypoint — all scene/physics/UI wiring lives in {@link mountStructureDemo}.
 */
import { mountStructureDemo } from './structure-demo.js';
import { buildFrameTowerScenario } from 'blast-stress-solver/scenarios';

mountStructureDemo({
  build: () => buildFrameTowerScenario(),
  label: 'Multi-storey frame tower',
});
