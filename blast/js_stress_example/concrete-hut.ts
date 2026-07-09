/**
 * "Mini concrete hut" destructible demo (vibe-city /destructible-stress preset).
 * Thin entrypoint — all scene/physics/UI wiring lives in {@link mountStructureDemo}.
 */
import { mountStructureDemo } from './structure-demo.js';
import { buildConcreteHutScenario } from 'blast-stress-solver/scenarios';

mountStructureDemo({
  build: () => buildConcreteHutScenario(),
  label: 'Mini concrete hut',
  // The hut is small and light — a gentler default shot reads better.
  projectile: { radius: 0.3, mass: 1_500, speed: 45 },
});
