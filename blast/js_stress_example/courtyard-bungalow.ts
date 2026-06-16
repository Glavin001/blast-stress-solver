/**
 * "Courtyard bungalow" destructible demo (vibe-city /destructible-stress preset).
 * Thin entrypoint — all scene/physics/UI wiring lives in {@link mountStructureDemo}.
 */
import { mountStructureDemo } from './structure-demo.js';
import { buildCourtyardBungalowScenario } from 'blast-stress-solver/scenarios';

mountStructureDemo({
  build: () => buildCourtyardBungalowScenario(),
  label: 'Courtyard bungalow',
});
