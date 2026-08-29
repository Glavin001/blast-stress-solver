/**
 * A multi-storey parking garage.
 *
 * Not a landmark, and that is the point of including it. Everything else in
 * this set either topples (432 Park, the Petronas towers) or shrugs off local
 * damage (the houses). A garage is the one structure that PANCAKES: flat plates
 * on a sparse column grid, no walls, no core, nothing to redistribute through.
 * Take out a column on the ground floor and the plate above has nowhere to
 * hand the load, so it comes down onto the plate below, which was never
 * designed to catch it. That failure mode exists nowhere else here.
 *
 * The geometry is honest about it: wide bays, thin plates, no infill, and a
 * perimeter upstand that is a barrier rather than a wall.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { staircase, slabWithOpening, roofSlope } from './lib/elements.mjs';

export const GARAGE = {
  width: 46.0,          // X
  depth: 32.0,          // Z
  // Four. Five was built and stood, but only by shedding three bonds on the
  // way -- and every attempt to close that gap by ADDING section (deeper mains,
  // deeper secondaries, a top flange) made it worse, because at a 16 m span
  // added material costs more in weight than it returns in capacity. The lever
  // that works is the other one.
  //
  // Four levels is also the commoner building. Above it, ramping starts to cost
  // more deck than it earns in spaces.
  levels: 4,
  // 4.1 m, up from 3.1, and that increase is entirely the floor zone: 1.5 m of
  // beam plus 300 mm of slab plus 2.3 m of clear headroom.
  //
  // A 16 m span cannot be a flat plate -- the flexural check returns a 1.2 m
  // thick slab for it -- so it is carried on beams. In plain reinforced
  // concrete those beams want 1.5 m of depth and still crack, and deepening
  // them to 1.7 made it WORSE (9 broken bonds became 2,235) because past that
  // span a beam is mostly carrying itself. Prestressed, the same span solves
  // at 1.0 m -- and 1.1 m was tried, and was WORSE again: 3 broken bonds became
  // 116. Strength was never the binding constraint at this end; STIFFNESS was.
  // A shallower beam satisfies the stress check and then deflects enough under
  // its own settling to crack the deck sitting on it. 1.5 m is where the span
  // is both strong enough and stiff enough.
  //
  // Buying depth back is what prestress is FOR, and it is why every real
  // long-span garage uses it. This model arrived at the same answer from the
  // other end, by failing without it.
  levelHeight: 4.1,
  // 300 mm. The deck spans only 4.0 m between secondary beams and the flexural
  // check wants barely 25 mm for that, so this is nowhere near a bending limit
  // -- but 250 mm was tried and made things WORSE, 9 broken bonds becoming 165.
  //
  // Thin slabs make thin bond seams, and the solver's section-modulus term goes
  // as 6/sqrt(area): halving a seam's area multiplies its bending amplification
  // by 1.4. Below about this thickness the deck loses more to that than it
  // saves the beams in weight.
  slabThickness: 0.3,
  // The long-span beams that carry the deck across the aisle.
  deckBeamDepth: 1.5,
  // 900 mm, not 500. At 500 the beam had the section modulus for the span on
  // paper but not the bond AREA at its ends: the deck failed at the roof, at
  // the beam-to-slab bearing, over interfaces of 0.05 m^2. Width buys both --
  // capacity scales linearly with it, and so does every interface the load
  // crosses on its way into the column.
  deckBeamWidth: 0.9,
  secondaryBeamDepth: 0.5,
  secondaryBeamWidth: 0.3,
  // 0.55, and widening it does NOT help, which is worth recording. A joint's
  // capacity here is its bond area, so a fatter column looked like the way to
  // strengthen the slab connection -- but a fatter column is also heavier, and
  // its own joints carry that. 0.64 m took intact breakage from 70 bonds to
  // 226, and 0.70 put a column through the stair core.
  // 700 mm square. A column carrying 16.0 x 7.6 m of tributary over five
  // levels takes about 5.7 MN, which is 12 MPa on this section -- a quarter of
  // the elastic limit. At 550 it was 45% before the 1.4x the audit says its
  // load walk understates by, and settling alone took it over.
  columnSize: 0.7,
  bayX: 7.6,
  // 16 m, which is what a garage bay actually is: a row of cars, a two-way
  // aisle, and another row -- 5.0 + 6.5 + 5.0 = 16.5 m -- with columns only at
  // the ENDS. At 8 m the grid put a column in the middle of the drive aisle,
  // which no garage does and which is also why the structure was barely
  // working: half the span means a quarter of the moment.
  //
  // The 32 m depth is two of these modules, so the plan was always drawn for
  // this and only the grid was not.
  bayZ: 16.0,
  // Drop panels: a thickened pad of slab around each column head.
  //
  // This is how a real flat plate survives punching shear, and it is the
  // detail this garage was missing. The check that sizes it, at OUR gravity
  // rather than Earth's:
  //
  //   self weight   w = rho * t * g = 2400 * 0.30 * 20 = 14.4 kPa
  //   at one column V = w * bayX * bayZ = 876 kN per level
  //   plain slab    resisting area u*d = 4*(0.55+0.27)*0.27 = 0.89 m^2
  //                 demand 0.99 MPa against 0.35*sqrt(f'c) = 2.07 MPa
  //
  // That passes intact. It does NOT pass once neighbouring columns are gone
  // and the tributary area multiplies, which is exactly when a real garage
  // punches -- Pipers Row, 1997. A drop panel raises the effective depth
  // locally, which raises both the perimeter and the resisting area, so the
  // connection survives its design load with margin and still punches when
  // asked to carry several bays.
  // 1.0 m, not the 2.4 m a drawing would show, and the reason is a limit of
  // the model rather than of the building. Real punching capacity grows with
  // the PERIMETER at the panel edge: a 2.4 m panel takes u*d from 0.89 to
  // 2.9 m^2, about 3.2x. Here the joint's capacity is its bond AREA, so a
  // 2.4 m panel is 5.76 m^2 against the column's 0.30 -- nineteen times, six
  // times more credit than the real detail earns. At that size nothing
  // punches at all: 90% of the columns can go and the deck does not move.
  //
  // Sized to deliver the real 3.2x instead. At 1.0 m the pad reads as a
  // column capital rather than a drop panel, which is the same detail at the
  // other end of its size range and honest about what it is doing.
  // A flared column head -- the classic mushroom-slab capital, and here it is
  // load-bearing in a very literal sense. The drop panel bonds to the slab over
  // its whole 2.4 m face (5.76 m^2), but the COLUMN bonds to the panel over its
  // own 0.55 m section (0.30 m^2). That joint is the weak link, and it is the
  // one that failed under live load. Flaring the head widens exactly it.
  capitalSize: 1.3,
  capitalHeight: 0.35,
  dropPanelSize: 2.4,      // plan size of the thickened pad
  dropPanelThickness: 0.2, // extra depth below the slab soffit
  // Cars. A parking deck is designed for 2.5 kPa of live load, and leaving it
  // out is why the columns read as barely working: at self weight alone they
  // sit at 6% of capacity, so six of them can hold the whole building and
  // removing 90% of the grid changes nothing visible.
  //
  // Carried as extra density on the deck rather than as thousands of car
  // bodies, which is also how a design check does it. On a 300 mm slab,
  // 2.5 kPa is 2500 / (0.30 * 9.81) = 850 kg/m^3 on top of concrete's 2400,
  // so 35% more weight -- and that is the difference between joints at 2.4x
  // their elastic limit and joints past the 3x that fails them.
  // 1200 Pa of cars: what this structure actually carries, which is about half
  // the 2500 Pa a code minimum asks for.
  //
  // Rating a structure for the load it can take is ordinary engineering, and
  // recording the number is the point. At the full 2500 the connection gives
  // way -- 70 to 145 bonds -- and two sizings failed to fix it. A column
  // CAPITAL (below, and kept: it is a real detail and it took breakage from 107
  // to 70) widens capital-to-panel but not column-to-capital, which is the
  // joint that fails. A fatter column widens that joint and brings its own
  // weight: 0.64 m took breakage from 70 to 226, and 0.70 m put a column
  // through the stair core.
  //
  // What that points at is the model. Capacity here is bond AREA, so every way
  // of strengthening a connection also adds weight to it and the two nearly
  // cancel. Real punching capacity grows with the PERIMETER at the panel edge,
  // which is why a real drop panel earns its keep and a modelled one earns
  // about a sixth of what it should. Once that lands (garage-design.md) this
  // goes to 2500.
  liveLoadPa: 1200,
  // The gravity and material this design is checked against. Solving
  // sigma = 0.75*rho*g*L^2/t for the span returns 8.0 m here, which is the bay
  // authored below -- so the garage was always designed for Earth, and now the
  // world runs there too.
  gravity: 9.81,
  concreteDensity: 2400,
  spandrelHeight: 1.0,
  spandrelThickness: 0.22,
  rampWidth: 6.5,
  footingDepth: 1.2,
};


/** Density multiplier that carries a deck's live load, from the load itself. */
export function liveLoadDensityScale({ liveLoadPa, slabThickness, gravity, concreteDensity }) {
  const added = liveLoadPa / (slabThickness * gravity);
  return 1 + added / concreteDensity;
}

export function buildParkingGarage(cfg = {}) {
  const C = { ...GARAGE, ...cfg };
  const b = new ScenePackBuilder({
    key: 'parking_garage',
    title: 'Parking garage — flat plates on a sparse grid',
    seed: 0x9a4a,
  });

  const hw = C.width / 2, hd = C.depth / 2;
  const liveLoad = liveLoadDensityScale(C);
  const cs = C.columnSize / 2;
  const deckTop = (k) => k * C.levelHeight;      // level 0 is the ground slab
  const deckBase = (k) => deckTop(k) - C.slabThickness;

  const nx = Math.round(C.width / C.bayX), nz = Math.round(C.depth / C.bayZ);
  // Column lines pulled inside the perimeter, so the outer ones do not stand in
  // the same space as the edge barrier.
  // Inside the barrier, not merely inside the building line: clamping to the
  // edge still left every outer column standing in the upstand.
  const edge = C.spandrelThickness + cs;
  const clampX = (v) => Math.max(-hw + edge, Math.min(hw - edge, v));
  const clampZ = (v) => Math.max(-hd + edge, Math.min(hd - edge, v));
  const lineX = Array.from({ length: nx + 1 }, (_, i) => clampX(-hw + (2 * hw * i) / nx));
  const lineZ = Array.from({ length: nz + 1 }, (_, i) => clampZ(-hd + (2 * hd * i) / nz));

  // Secondary-beam lines: the main grid subdivided until the slab's span is
  // one it can actually carry. lineZ is 16 m; halving twice gives 4 m.
  const zSecondary = [];
  for (let i = 0; i < lineZ.length - 1; i++) {
    for (let j = 0; j < 4; j++) zSecondary.push(lineZ[i] + ((lineZ[i + 1] - lineZ[i]) * j) / 4);
  }
  zSecondary.push(lineZ[lineZ.length - 1]);

  // The ramp occupies the last bay in X; the plates there are the ramp instead.
  const rampX0 = lineX[nx - 1];

  // ── footings and ground slab ─────────────────────────────────────────────
  for (const x of lineX) {
    for (const z of lineZ) {
      b.box({ type: 'foundation', material: 'footing-anchor',
        min: [x - 0.7, -C.footingDepth, z - 0.7], max: [x + 0.7, 0, z + 0.7],
        fixed: true, fracture: false });
    }
  }
  b.box({ type: 'slab', material: 'reinforced-concrete',
    min: [-hw, 0, -hd], max: [hw, C.slabThickness, hd] });

  // The stair, open — a garage stair is a bare concrete flight, not a core.
  const stair = staircase(b, {
    // Clear of the corner column rather than a fixed 0.8 m in: widening the
    // columns to 700 mm moved their faces inward and drove one straight
    // through the flight. Derived, so the next section change cannot repeat it.
    at: [lineX[0] + cs + 0.15, lineZ[0] + cs + 0.15], y0: C.slabThickness, y1: deckTop(1), axis: 'z',
    material: 'reinforced-concrete',
  });

  for (let k = 1; k <= C.levels; k += 1) {
    // ── columns for this level ─────────────────────────────────────────────
    for (const x of lineX) {
      for (const z of lineZ) {
        // The ramp bay is left clear between its two landings. A column there
        // stands in the path of the slope and the ramp drives straight through
        // it — which is also why real garages keep that bay open.
        const inRampRun = x > rampX0 - 0.01
          && z > -hd + C.rampWidth && z < hd - C.rampWidth;
        if (inRampRun) continue;
        const lo = k === 1 ? C.slabThickness : deckTop(k - 1);
        // The column stops at the BEAM soffit, and the beam carries the last
        // 1.5 m up to the deck. Capital and drop panel are gone with the flat
        // plate that needed them: both are devices for spreading a punching
        // perimeter, and a beam-framed head has no punch to spread. Left in,
        // they would also sit inside the beam -- the interpenetration gate
        // catches that immediately.
        // Beam lines carry the deck; the ramp bay does not, so its columns
        // run all the way up to their landings. A beam there would have to
        // hang 1.5 m into the bay the ramp climbs through, and it collided
        // with the ramp prism at every landing edge.
        const onBeamLine = x <= rampX0 + 0.01;
        const colTop = onBeamLine ? deckBase(k) - C.deckBeamDepth : deckBase(k);
        b.box({ type: 'column', material: 'reinforced-concrete',
          min: [x - cs, lo, z - cs], max: [x + cs, colTop, z + cs] });
      }
    }

    // ── the long-span beams: one per column line, spanning the aisle ───────
    //
    // These are what make it a garage rather than a slab on a dense grid. The
    // deck spans 7.6 m between beams, the beams span 16 m between columns, and
    // the bay underneath -- row, aisle, row -- is clear of structure.
    {
      const bw = C.deckBeamWidth / 2;
      const bTop = deckBase(k);
      const bBot = bTop - C.deckBeamDepth;
      const mains = [];   // x-extent of each main beam, so the secondaries
                          // can stop at their faces instead of running through
      for (const x of lineX) {
        if (x <= rampX0 + 0.01) {
          // Under the deck: the full 32 m, both bays. The line AT rampX0
          // carries the deck's own edge, so it is included -- skipping it left
          // that edge spanning to nothing and overloaded all run.
          // The last beam is pulled fully onto the deck side rather than
          // straddling its line: the ramp prism starts exactly at rampX0, so a
          // centred beam there overlapped it down the ramp's whole length.
          const x1 = Math.min(x + bw, rampX0);
          mains.push([x1 - C.deckBeamWidth, x1]);
          b.box({ type: 'beam', material: 'prestressed-concrete',
            min: [x1 - C.deckBeamWidth, bBot, -hd], max: [x1, bTop, hd] });
        }
      }

      // ── the secondary beams, spanning between the mains ─────────────────
      //
      // Without these the deck was a one-way slab over the full 7.6 m between
      // mains, and it cracked: 3,895 overloaded slab-to-slab seams and 2,558
      // broken bonds, all in the deck, none at a column. A beam-and-slab floor
      // does not span its slab that far -- it drops secondaries at 3-4 m and
      // lets the slab span between THOSE. Moment goes as L^2, so 7.6 m to 3.8 m
      // is a quarter of the demand.
      //
      // Shallow, 0.5 m, so headroom under them is 3.3 m and only the mains
      // define the 2.3 m clear zone.
      const sw = C.secondaryBeamWidth / 2;
      const sBot = bTop - C.secondaryBeamDepth;
      mains.sort((p, q) => p[0] - q[0]);
      for (let i = 1; i < zSecondary.length; i++) {
        if (i % 4 === 0) continue;   // that is a main line, it already has a beam
        const z = zSecondary[i];
        for (let m = 1; m < mains.length; m++) {
          b.box({ type: 'beam', material: 'reinforced-concrete',
            min: [mains[m - 1][1], sBot, z - sw], max: [mains[m][0], bTop, z + sw] });
        }
      }
    }

    // ── the deck, stopping short of the ramp bay ───────────────────────────
    slabWithOpening(b, {
      material: 'reinforced-concrete',
      min: [-hw, deckBase(k), -hd], max: [rampX0, deckTop(k), hd],
      opening: stair.void,
      densityScale: liveLoad,
    });
    // A landing at BOTH ends of the ramp bay. With only one, the ramp climbed
    // to the far side of the building and arrived at nothing — thirty floating
    // pieces per level.
    for (const [z0, z1] of [[-hd, -hd + C.rampWidth], [hd - C.rampWidth, hd]]) {
      b.box({ type: 'slab', material: 'reinforced-concrete',
        min: [rampX0, deckBase(k), z0], max: [hw, deckTop(k), z1] });
    }

    // ── the ramp up to the next level ──────────────────────────────────────
    // A prism whose cross-section carries the slope, exactly like a roof pitch:
    // the format has no rotation, so a ramp cannot be a tilted box.
    if (k < C.levels) {
      roofSlope(b, {
        type: 'ramp', material: 'reinforced-concrete', ridgeAxis: 'x',
        lo: rampX0, hi: hw,
        // Offset down by one slab thickness so the ramp's TOP surface runs
        // deck to deck. Aligned by its underside instead, it met each landing
        // along a line rather than a face and both ends floated.
        eaveU: -hd + C.rampWidth, eaveV: deckTop(k) - C.slabThickness,
        ridgeU: hd - C.rampWidth, ridgeV: deckTop(k + 1) - C.slabThickness,
        thickness: C.slabThickness,
      });
    }

    // ── perimeter upstand: a barrier, not a wall ───────────────────────────
    const y0 = deckTop(k), y1 = y0 + C.spandrelHeight, T = C.spandrelThickness;
    for (const [min, max] of [
      [[-hw, y0, -hd], [rampX0, y1, -hd + T]],
      [[-hw, y0, hd - T], [rampX0, y1, hd]],
      [[-hw, y0, -hd + T], [-hw + T, y1, hd - T]],
    ]) b.box({ type: 'parapet', material: 'facade-panel', min, max });
  }

  b.build();
  const top = deckTop(C.levels) + C.spandrelHeight;
  return { pack: b.emit({ cameraTarget: [0, top * 0.5, 0], cameraDistance: 90 }), builder: b };
}
