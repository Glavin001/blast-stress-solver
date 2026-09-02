/**
 * Sizing a framed floor: the checks, as functions.
 *
 * These existed already, but as throwaway `node -e` one-liners typed fresh each
 * time a member changed, and afterwards only in commit messages. That is fine
 * until gravity moves, or a span does, and then every number in every structure
 * is a magic constant whose derivation has to be reconstructed from prose.
 *
 * Everything here is a SERVICE-level check against cracking, not an ultimate
 * check: this project's materials crack at `tensionFrac * compressionElastic`
 * and go outright at the fatal multiple of that, and a structure that spends
 * its life above the first number is one the solver will slowly take apart.
 *
 * All SI. Gravity is a parameter everywhere, because it has already changed
 * once in this project and the buildings did not survive it.
 */

/** Distributed load on a deck, in Pa: its own weight plus what it carries. */
export function deckLoadPa({ thickness, density = 2400, gravity = 9.81, liveLoadPa = 0 }) {
  return thickness * density * gravity + liveLoadPa;
}

/**
 * Peak bending stress in a one-way slab of `thickness` spanning `span`.
 *
 * `sigma = 0.75 * w * L^2 / t^2` per unit width, with the 0.75 standing in for
 * a continuous rather than simply-supported span. Returns Pa.
 */
export function slabStress({ span, thickness, density = 2400, gravity = 9.81, liveLoadPa = 0 }) {
  const w = deckLoadPa({ thickness, density, gravity, liveLoadPa });
  return (0.75 * w * span * span) / (thickness * thickness);
}

/**
 * Thickness a one-way slab needs to stay under `allowable` over `span`.
 *
 * Solved by iteration rather than algebra because the slab's own weight is part
 * of the load it has to carry, so thickness appears on both sides.
 *
 * NOTE the floor. Below roughly 200 mm the binding constraint stops being
 * bending and becomes BOND AREA: the solver's section-modulus term goes as
 * 6/sqrt(area), so halving a seam's area multiplies its bending amplification
 * by 1.4. A 250 mm garage deck passes this check with two decimal places to
 * spare and still made things worse than a 300 mm one -- 9 broken bonds became
 * 165 -- because thin slabs make thin seams. Take the answer as a lower bound.
 */
export function slabThicknessFor({
  span, allowable, density = 2400, gravity = 9.81, liveLoadPa = 0, floor = 0.2,
}) {
  let t = 0.1;
  for (let i = 0; i < 200; i += 1) {
    const s = slabStress({ span, thickness: t, density, gravity, liveLoadPa });
    if (s <= allowable) break;
    t += 0.005;
  }
  return Math.max(t, floor);
}

/**
 * Peak bending stress in a beam of `width` x `depth` spanning `span` and
 * carrying `tributaryWidth` of deck. Returns Pa.
 *
 * `M = wL^2/8`, `S = bd^2/6`, both for a simply-supported span -- deliberately
 * conservative, because continuity at the supports is exactly what a chunked
 * bond model does NOT reliably provide.
 */
export function beamStress({
  span, width, depth, tributaryWidth, deckThickness,
  density = 2400, gravity = 9.81, liveLoadPa = 0,
}) {
  const deck = tributaryWidth * deckLoadPa({
    thickness: deckThickness, density, gravity, liveLoadPa,
  });
  const own = width * depth * density * gravity;
  const w = deck + own;
  const moment = (w * span * span) / 8;
  const modulus = (width * depth * depth) / 6;
  return moment / modulus;
}

/** The shallowest beam that stays under `allowable`, or null if none does. */
export function beamDepthFor({ allowable, maxDepth = 3.0, ...rest }) {
  for (let d = 0.2; d <= maxDepth; d += 0.01) {
    if (beamStress({ depth: d, ...rest }) <= allowable) return Math.round(d * 100) / 100;
  }
  return null;
}

/**
 * What share of a beam's total load is its own weight.
 *
 * The strength-limited / mass-limited diagnostic, at authoring time rather than
 * after a build. As this rises, a given percentage of extra depth returns less,
 * because depth adds weight linearly while it adds capacity quadratically only
 * against a load that is not itself growing.
 *
 * The parking garage's 16 m mains sit near a quarter. Deepening them from 1.5 m
 * to 1.7 m took broken bonds from 9 to 2,235.
 */
export function beamSelfWeightShare({
  width, depth, tributaryWidth, deckThickness,
  density = 2400, gravity = 9.81, liveLoadPa = 0,
}) {
  const own = width * depth * density * gravity;
  const deck = tributaryWidth * deckLoadPa({
    thickness: deckThickness, density, gravity, liveLoadPa,
  });
  return own / (own + deck);
}

/**
 * The widest spacing of secondary beams that keeps the deck under `allowable`.
 *
 * The deck spans between secondaries, so this is `slabThicknessFor` run
 * backwards: fix the slab, solve for the span.
 */
export function secondarySpacingFor({
  thickness, allowable, density = 2400, gravity = 9.81, liveLoadPa = 0, max = 12,
}) {
  let best = 0;
  for (let L = 0.5; L <= max; L += 0.1) {
    if (slabStress({ span: L, thickness, density, gravity, liveLoadPa }) <= allowable) best = L;
    else break;
  }
  return Math.round(best * 10) / 10;
}

/**
 * Throw unless a framed deck's authored members satisfy their checks.
 *
 * Call it from a structure so the design is verified at build time by the same
 * formulas that produced it, and a change to gravity, a span or a material
 * fails loudly at the source instead of silently in a simulation twenty minutes
 * later.
 *
 * Reports every failing check at once. Fixing them one build at a time is how
 * an afternoon disappears.
 */
export function checkFramedDeck(label, {
  deckSpan, deckThickness, deckAllowable,
  beamSpan, beamWidth, beamDepth, beamTributary, beamAllowable,
  density = 2400, gravity = 9.81, liveLoadPa = 0,
}) {
  const problems = [];
  const ds = slabStress({ span: deckSpan, thickness: deckThickness, density, gravity, liveLoadPa });
  if (ds > deckAllowable) {
    problems.push(`deck: ${(ds / 1e6).toFixed(2)} MPa over ${deckSpan} m at ${deckThickness} m `
      + `exceeds ${(deckAllowable / 1e6).toFixed(2)} MPa; needs `
      + `${slabThicknessFor({ span: deckSpan, allowable: deckAllowable, density, gravity, liveLoadPa })} m`);
  }
  const bs = beamStress({
    span: beamSpan, width: beamWidth, depth: beamDepth, tributaryWidth: beamTributary,
    deckThickness, density, gravity, liveLoadPa,
  });
  if (bs > beamAllowable) {
    const need = beamDepthFor({
      allowable: beamAllowable, span: beamSpan, width: beamWidth,
      tributaryWidth: beamTributary, deckThickness, density, gravity, liveLoadPa,
    });
    problems.push(`beam: ${(bs / 1e6).toFixed(2)} MPa over ${beamSpan} m at `
      + `${beamWidth}x${beamDepth} m exceeds ${(beamAllowable / 1e6).toFixed(2)} MPa; needs `
      + `${need === null ? 'more depth than is sane — prestress it or shorten the span' : `${need} m`}`);
  }
  if (problems.length) {
    throw new Error(`${label}: the framing does not check out\n  - ${problems.join('\n  - ')}`);
  }
  return {
    deckStress: ds,
    beamStress: bs,
    beamSelfShare: beamSelfWeightShare({
      width: beamWidth, depth: beamDepth, tributaryWidth: beamTributary,
      deckThickness, density, gravity, liveLoadPa,
    }),
  };
}
