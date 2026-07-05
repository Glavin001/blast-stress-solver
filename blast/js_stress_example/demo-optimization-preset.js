/**
 * Validated "recommended" settling preset for the bundled demos.
 *
 * Mirrors SLEEP_DAMPING_TREATMENTS.recommended from the sleep/damping A/B study
 * (blast-stress-solver/src/tests/stress/runSuite.ts). Keep the two in sync.
 *
 * Findings from that study (small tier, WASM):
 *   - Aggressive sleep everywhere is safe. A 0.5 linear/angular threshold only bites
 *     near-zero velocity, so airborne flight is untouched, while a settled scene's
 *     steady-state tail gets up to ~3.5x cheaper and final-frame awake bodies drop
 *     ~5x (41 -> 8). This is the fix for "rubble never sleeps".
 *   - Damping is gated to 'afterGroundCollision' (post-landing only). Damping a piece
 *     mid-air caps it at terminal velocity (~g/damping) and reads as a floaty,
 *     slow-motion collapse, so airborne pieces are never damped — peak airborne speeds
 *     match the un-damped baseline.
 *
 * This is applied in the demo path ONLY. The library default
 * (buildDestructibleCore: sleepMode 'off', smallBodyDamping 'off') is intentionally
 * left unchanged so library consumers opt in explicitly.
 */

/** @typedef {'off' | 'always' | 'afterGroundCollision'} OptimizationMode */

/**
 * Sleep settings — each demo spreads this at the top level of its
 * buildDestructibleCore options.
 * @type {{ sleepMode: OptimizationMode; sleepLinearThreshold: number; sleepAngularThreshold: number }}
 */
export const RECOMMENDED_SLEEP = {
  sleepMode: 'always',
  sleepLinearThreshold: 0.5,
  sleepAngularThreshold: 0.5,
};

/**
 * Damping thresholds — spread into the `smallBodyDamping` object by
 * physicsCoreOverrides() in physics-controls.ts. The damping *mode* stays a UI
 * dropdown there (shared physicsConfig state, default 'off'), so only the numeric
 * tuning lives here; it takes effect when the dropdown enables damping.
 * @type {{ colliderCountThreshold: number; minLinearDamping: number; minAngularDamping: number }}
 */
export const RECOMMENDED_DAMPING = {
  colliderCountThreshold: 8,
  minLinearDamping: 0.6,
  minAngularDamping: 0.6,
};
