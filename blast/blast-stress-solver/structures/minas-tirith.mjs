/**
 * Minas Tirith — the seven-walled city of Gondor.
 *
 * Seven concentric tiers cut into a mountain spur, a great gate facing east, a
 * road that switchbacks up through tunnels in the rock, and the Citadel with
 * its prow of stone jutting out over the plain.
 *
 * Not to scale, and by a long way. The city is canonically some 700 m across
 * with 30 m walls and a 200 m spur; at this library's fracture density a single
 * ring wall of that size is ~200,000 chunks. Modelled at roughly 1:3 — 240 m
 * across, 12 m per tier — which keeps the seven rings that make the silhouette
 * legible while landing in the same chunk budget as the other packs.
 *
 * What makes this different from every other structure here: it is a place you
 * walk through, so geometry is sized against the player capsule, not just
 * against what looks right. See PLAYER below.
 */
import { ScenePackBuilder } from './lib/pack.mjs';
import { ringSegments, ringPoint } from './lib/shapes.mjs';

/**
 * The player, from netcode/src/movement.rs. Every clearance here is derived
 * from these and nothing is authored tighter.
 *
 * The capsule is 1.6 m tall and 0.7 m across and CROUCH DOES NOT SHRINK IT, so
 * 1.6 m is the hard ceiling everywhere. Steps up to 0.55 m are climbed
 * automatically, but bots are configured at 0.35 m, so risers stay under that.
 * Slopes auto-slide above 30 deg. A jump clears 1.05 m, which is why parapets
 * are 1.2 m — high enough that you cannot hop off the wall by accident.
 */
const PLAYER = {
  height: 1.6, width: 0.7, step: 0.35, jump: 1.05,
  door: { w: 1.4, h: 2.4 },   // generous: 0.72 x 1.62 is the hard floor and feels awful
  walkway: 2.0,               // clear width; below ~1.5 the snap-to-ground feels like falling
  headroom: 3.0,              // tunnels and arcades
};

export const MINAS_TIRITH = {
  tiers: 7,
  outerRadius: 120,
  tierDrop: 16,        // radius step per tier
  tierHeight: 16,      // rise per tier; 7 x 16 = 112 m against 240 m across
  cliffThickness: 3.5, // the rock face below each terrace; the wall stands on its top
  deckThickness: 1.5,
  /** Terrain cell size. See the note on `cellVolume` in pack.mjs. */
  rockCell: 120,
  roadWidth: 6,     // leaves ~7 m of the 16 m terrace for houses
  roadArc: Math.PI / 3,   // 60 deg per tier: ~11% grade, a comfortable walk
  roadRise: 0.25,         // per step; under the 0.35 m bot climb limit
  spurHalfWidth: 7,
  segments: 48,     // shared by every ring; 15.7 m chord at r=120
  // Wall height and thickness taper inwards, as they do in the reference: the
  // outer wall is the great one you walk along, and the inner rings read as
  // ramparts between the buildings rather than fortifications in their own
  // right. This is also the chunk budget's main dial -- stone fractures to
  // masonry scale (~1 chunk per m^3), so wall VOLUME is very nearly the node
  // count, and at a uniform 9 m x 3.5 m the seven rings alone came to 70,000.
  wallByTier: [
    { h: 8, t: 3.0 },   // tier 1: the great wall, walkable
    { h: 5, t: 1.5 }, { h: 5, t: 1.5 }, { h: 5, t: 1.5 },
    { h: 3.5, t: 1.5 }, { h: 3.5, t: 1.5 }, { h: 3.5, t: 1.5 },
  ],
  parapet: 1.2,     // taller than the 1.05 m jump, so you cannot hop off a rampart
  merlonsPerSegment: 8,
};

const deg = (d) => (d * Math.PI) / 180;

export function buildMinasTirith(cfg = {}) {
  const C = { ...MINAS_TIRITH, ...cfg };
  const b = new ScenePackBuilder({ key: 'minas_tirith', title: 'Minas Tirith — the White City' });

  // Tier k (1-based) has its wall at radius r(k) and its terrace at height y(k).
  const r = (k) => C.outerRadius - C.tierDrop * (k - 1);
  const y = (k) => C.tierHeight * (k - 1);
  // `footing-anchor`, not `stone`. It is what the mountain IS -- pinned
  // foundation -- and it also reads correctly: `stone` resolves to a texture
  // whose mean albedo is 0.11 linear, which rendered the rock almost black
  // against the white city instead of grey.
  const rock = { material: 'footing-anchor', fixed: true, fracture: false, cellVolume: C.rockCell };

  // ── the mountain ─────────────────────────────────────────────────────────
  //
  // A terraced shell, never a solid mass. The subdivision grid applies whether
  // or not a piece fractures, so a solid cone at this size is over a million
  // nodes; as a shell with a raised cell it is a few hundred. Nothing sees the
  // inside.
  //
  // Each tier is a deck plus the cliff below the NEXT tier in. The cliff starts
  // one deck-thickness low so it shares a real vertical face with the deck
  // outside it rather than meeting it along a line, which would carry no bond.
  // One segment count for every ring, not one per tier.
  //
  // A deck's inner edge and the cliff face inboard of it meet at the same
  // nominal radius, but a ring segment is a straight chord — so its boundary
  // dips inside that radius everywhere except at the vertices. Two rings
  // faceted differently therefore interpenetrate, and 280 pairs of them did.
  // Sharing the segmentation makes the chords coincide exactly.
  const SEG = C.segments;

  for (let k = 1; k <= C.tiers; k += 1) {
    const seg = SEG;

    // The cliff face below this tier's terrace. Tier 1 stands on the plain.
    if (k > 1) {
      for (const s of ringSegments({
        rInner: r(k) - C.cliffThickness, rOuter: r(k), segments: seg,
      })) {
        b.group(`rock-t${k}-${s.i >> 3}`, () => b.piece({ type: 'foundation', ...rock,
          axis: 'y', poly: s.poly, lo: y(k - 1) - C.deckThickness, hi: y(k) }));
      }
    }

    // The terrace deck, inboard of the cliff top. The cliff's own top face is
    // flush with it, so the walkable surface runs unbroken from the deck's
    // inner edge out to r(k).
    const deckOuter = k === 1 ? r(k) : r(k) - C.cliffThickness;
    if (k === C.tiers) {
      // The Citadel floor is a full disc, so it is wedges from the centre
      // rather than an annulus with something plugging the hole. A separate
      // hub cannot close that hole cleanly: an octagon's vertices reach past
      // the ring's inner chord and overlap it, a smaller one leaves a crack.
      for (let i = 0; i < seg; i += 1) {
        const t0 = (2 * Math.PI * i) / seg, t1 = (2 * Math.PI * (i + 1)) / seg;
        b.group(`deck-t${k}-${i >> 3}`, () => b.piece({ type: 'foundation', ...rock,
          axis: 'y', lo: y(k) - C.deckThickness, hi: y(k),
          poly: [[0, 0], ringPoint(0, 0, deckOuter, t0), ringPoint(0, 0, deckOuter, t1)] }));
      }
      continue;
    }
    for (const s of ringSegments({ rInner: r(k + 1), rOuter: deckOuter, segments: seg })) {
      b.group(`deck-t${k}-${s.i >> 3}`, () => b.piece({ type: 'foundation', ...rock,
        axis: 'y', poly: s.poly, lo: y(k) - C.deckThickness, hi: y(k) }));
    }
  }

  // ── the road ─────────────────────────────────────────────────────────────
  //
  // A sloping curved ramp is not expressible: an `axis:'y'` wedge has a flat
  // top and a flat bottom, and there is no rotation to tilt one with. So the
  // road climbs in steps — one annular wedge per step, each filled solid down
  // to the terrace, so it reads as a ramp built against the cliff rather than
  // a ledge floating off it.
  //
  // The step is 0.25 m, under the 0.35 m a bot will climb and well under the
  // player's 0.55 m, so it walks like a slope rather than a stair. 60 degrees
  // of arc per tier puts the grade near 11%.
  //
  // Direction alternates: tier 1 climbs anticlockwise to +30 deg, tier 2 climbs
  // back clockwise from there. That is a real switchback — you arrive at the
  // foot of the next road rather than having to walk the terrace back round.
  const roadSteps = Math.ceil(C.tierHeight / C.roadRise);
  for (let k = 1; k < C.tiers; k += 1) {
    const ccw = k % 2 === 1;
    const from = ccw ? -C.roadArc / 2 : C.roadArc / 2;
    const to = ccw ? C.roadArc / 2 : -C.roadArc / 2;
    // Held off the cliff by 6 cm. The road is finely segmented and the cliff is
    // not, so their chords do not coincide; touching nominal radii would
    // interpenetrate. All of this is pinned, so nothing is lost by not bonding.
    const rIn = r(k + 1) + 0.06;
    ringSegments({ rInner: rIn, rOuter: rIn + C.roadWidth, segments: roadSteps, from, to })
      .forEach((seg, i) => {
        b.group(`road-t${k}-${i >> 3}`, () => b.piece({ type: 'ramp', ...rock, axis: 'y',
          poly: seg.poly,
          lo: y(k),   // ON the terrace, not into it
          hi: y(k) + (C.tierHeight * (i + 1)) / roadSteps }));
      });
  }

  // ── the spur ─────────────────────────────────────────────────────────────
  //
  // The blade of rock that splits the city, and the reason the road tunnels.
  // It stands on each terrace from tier 2 inwards; tier 1 is left clear so the
  // ground inside the Great Gate is open.
  //
  // Where the road crosses it, the rock is authored only ABOVE the tunnel
  // ceiling — the tunnel is the gap between the road's surface and the rock
  // over it, which is the only way to express a hole. A chunk's collider is its
  // convex hull, so a tunnel bored through a single piece would be filled back
  // in and the player would walk into it.
  const hw = C.spurHalfWidth;
  for (let k = 2; k < C.tiers; k += 1) {
    // Kept inside the terrace AND clear of the wall standing on its outer lip.
    // At z = +/-hw a point at x sits at radius hypot(x, hw), so the limit comes
    // off the wall's inner face, not the tier radius — measuring to the tier
    // radius ran the blade through the tier 6 wall.
    const xOut = Math.sqrt((r(k) - C.wallByTier[k - 1].t) ** 2 - hw ** 2) - 0.15;
    const xTunnel = r(k + 1) + C.roadWidth + 0.5;
    const top = k === C.tiers - 1 ? y(C.tiers) : y(k + 1) + 8;
    const zPoly = [[xTunnel, -hw], [xOut, -hw], [xOut, hw], [xTunnel, hw]];
    b.group(`spur-t${k}`, () => b.piece({ type: 'foundation', ...rock, axis: 'y',
      poly: zPoly, lo: y(k), hi: top }));
    // Over the tunnel mouth. The road crosses the spur at the midpoint of its
    // climb, so its surface here is half a tier up; the ceiling clears that by
    // the headroom, which is well over the 1.6 m the player needs.
    //
    // How much the road climbs while inside the spur is not a constant: the
    // spur is a fixed width in metres, so at the inner tiers it spans a much
    // wider ANGLE, and the road gains far more height crossing it. Assuming
    // half a tier put the ceiling through the road at tier 6.
    const halfAngle = Math.asin(Math.min(1, hw / r(k + 1)));
    const climbInside = (halfAngle / C.roadArc) * C.tierHeight;
    const ceiling = y(k) + C.tierHeight / 2 + climbInside + PLAYER.headroom;
    if (top > ceiling + 1) {
      b.group(`spur-t${k}`, () => b.piece({
        type: 'foundation', ...rock, axis: 'y', lo: ceiling, hi: top,
        poly: [[r(k + 1), -hw], [xTunnel, -hw], [xTunnel, hw], [r(k + 1), hw]],
      }));
    }
  }

  // ── the walls ────────────────────────────────────────────────────────────
  //
  // One ring per tier, standing on the cliff top (tier 1 stands on the ground
  // inside it), each with a gate where the road arrives. The ring is offset so
  // that a whole segment is centred on the gate: that segment is then skipped
  // and rebuilt as jambs and a lintel, because a chunk's collider is its convex
  // hull and an archway punched through one piece would be filled solid again.
  const wallRing = (tier, gates, height, thickness, gateW, gateH) => {
    const SEG_W = C.segments;
    const half = Math.PI / SEG_W;
    const rOut = r(tier);
    const rIn = rOut - thickness;
    const base = y(tier);
    const top = base + height;

    // How many segments the gateway swallows. A fixed gate width is a WIDER
    // angle the further in you go, and at tier 7 a 4.5 m opening is broader
    // than one 48th of the ring — skipping a single segment there inverted the
    // jamb's angular range and folded it back through the wall. Segment 0 is
    // centred on the gate, so dropping i <= j and i >= SEG-j stays symmetric.
    const dA = gateW / 2 / rIn;
    const j = Math.max(0, Math.ceil((dA / half - 1) / 2));
    const halfSpan = (2 * j + 1) * half;
    const gateAngle = gates[0];
    // Any further gates are matched by angle rather than by index, since the
    // ring is only offset to centre a segment on the first one.
    const norm = (a) => Math.atan2(Math.sin(a), Math.cos(a));
    const inAGate = (mid) => gates.some((g) => Math.abs(norm(mid - g)) < halfSpan - 1e-6);

    ringSegments({
      rInner: rIn, rOuter: rOut, segments: SEG_W,
      from: gateAngle - half, to: gateAngle - half + Math.PI * 2,
      skip: (_i, t0, t1) => inAGate((t0 + t1) / 2),
    }).forEach((seg) => {
      b.group(`wall-t${tier}-${seg.i >> 3}`, () => b.piece({ type: 'wall',
        material: 'white-stone', axis: 'y', poly: seg.poly, lo: base, hi: top }));
    });

    // The gateway, as wedges of the same ring rather than boxes.
    //
    // A box would have to be rotated to sit in a gate that is not on an axis,
    // and the format has no rotation — an earlier pass authored the jambs on
    // +x regardless of where the gate was, which put them straight through the
    // wall at every tier above the first. Expressed in angle instead, a jamb is
    // just a shorter segment and its radial faces still line up exactly with
    // the neighbours', so it bonds to them.
    const wedge = (from, to, y0, y1) => {
      ringSegments({ rInner: rIn, rOuter: rOut, segments: 1, from, to }).forEach((seg) => {
        b.piece({ type: 'wall', material: 'white-stone', axis: 'y', poly: seg.poly, lo: y0, hi: y1 });
      });
    };
    for (const g of gates) {
      wedge(g - halfSpan, g - dA, base, base + gateH);            // jamb
      wedge(g + dA, g + halfSpan, base, base + gateH);            // jamb
      wedge(g - halfSpan, g + halfSpan, base + gateH, top);       // lintel
    }

    // Crenellations: merlons on the outer lip, gaps between them. Held inside
    // the wall's own footprint by taking the nominal radius the wall's straight
    // chords actually reach at their midpoint, not the radius of its vertices.
    const rMerlonOut = rOut * Math.cos(half);
    ringSegments({
      rInner: rMerlonOut - 0.8, rOuter: rMerlonOut,
      segments: SEG_W * C.merlonsPerSegment,
      from: gateAngle - half, to: gateAngle - half + Math.PI * 2,
      skip: (i) => i % 2 === 1,
    }).forEach((seg) => {
      b.group(`merlon-t${tier}-${seg.i >> 5}`, () => b.piece({ type: 'parapet',
        material: 'white-stone', axis: 'y', poly: seg.poly, lo: top, hi: top + C.parapet }));
    });
  };

  // Tier 1's gate faces east and is the Great Gate. Every tier above takes its
  // gate where that tier's road arrives, which alternates with the switchback.
  const arrivalAngle = (m) => ((m - 1) % 2 === 1 ? C.roadArc / 2 : -C.roadArc / 2);
  wallRing(1, [0], C.wallByTier[0].h, C.wallByTier[0].t, 6, 6.5);
  for (let m = 2; m <= C.tiers; m += 1) {
    const w = C.wallByTier[m - 1];
    // The Citadel takes a second opening due east, which is the only way out
    // onto the prow.
    const gates = m === C.tiers ? [arrivalAngle(m), 0] : [arrivalAngle(m)];
    wallRing(m, gates, w.h, w.t, 4.5, Math.min(3.2, w.h - 0.8));
  }

  // ── the town ─────────────────────────────────────────────────────────────
  //
  // Houses follow the ring rather than sitting square on it: a wedge-shaped
  // plan needs no rotation, which the format does not have, and terraced
  // buildings on a curved street is what the reference shows anyway.
  //
  // Hollow, four walls and a roof. Solid blocks would cost several times the
  // chunks for something you can only see the outside of, and a hollow one can
  // be given a door later without changing anything else.
  const house = ({ angle, arcWidth, rIn, rOut, base, height, doorAt = null }) => {
    const t0 = angle - arcWidth / 2, t1 = angle + arcWidth / 2;
    const wallT = 0.55;
    const top = base + height;
    // A face is a chord, and a chord bows inside the arc it spans. Subdivide
    // on the SAG rather than on the angle: a fixed 0.1 rad step is harmless at
    // the Hall's radius but bows 0.14 m at the outer tier, which is far past
    // the 5 mm interpenetration limit and put the street face straight through
    // the cross beam sitting on that very line.
    const maxSag = 0.003;
    const step = 2 * Math.acos(Math.max(-1, Math.min(1, 1 - maxSag / rOut)));
    const arcSegs = (a) => Math.max(1, Math.ceil(Math.abs(a) / step));
    const put = (rA, rB, from, to, lo, hi, type = 'wall', mat = 'white-stone') => {
      if (!(to > from) || !(rB > rA)) return;
      ringSegments({ rInner: rA, rOuter: rB, segments: arcSegs(to - from), from, to })
        .forEach((seg) => b.piece({ type, material: mat, axis: 'y', poly: seg.poly, lo, hi }));
    };

    // The side walls take the full depth and own the corners; the street and
    // back faces stop short of them. Sharing an exact radial boundary is what
    // makes these bond -- overlapping them instead (the obvious way round) put
    // every corner over the 5 mm interpenetration limit.
    const dRad = wallT / ((rIn + rOut) / 2);
    put(rIn, rOut, t0, t0 + dRad, base, top);
    put(rIn, rOut, t1 - dRad, t1, base, top);

    const fa = t0 + dRad, fb = t1 - dRad;
    put(rOut - wallT, rOut, fa, fb, base, top);          // back, against the hill
    if (doorAt === 'inner') {
      // The door is the gap between two piers, never a hole in one piece: a
      // chunk's collider is its convex hull and would fill an arch back in.
      const dA = PLAYER.door.w / 2 / rIn;
      put(rIn, rIn + wallT, fa, angle - dA, base, top);
      put(rIn, rIn + wallT, angle + dA, fb, base, top);
      put(rIn, rIn + wallT, angle - dA, angle + dA, base + PLAYER.door.h, top);
    } else {
      put(rIn, rIn + wallT, fa, fb, base, top);
    }
    // Timber, not stone, and for a structural reason rather than a decorative
    // one. A roof spans the whole house unsupported, and 0.3 m of stone over
    // 7 m develops about 3.1 MPa in bending against stone's 0.8 MPa tension
    // limit -- authored in stone, 1,707 of 3,613 roof chunks tore off and fell
    // under gravity alone. Timber is both four times lighter and five times
    // stronger in tension, which puts the same span at 0.7 MPa. It is also
    // what actually roofed these buildings.
    // A ridge beam across the middle, carried on the side walls. Timber alone
    // still leaves a 7 m span right at its limit -- 141 roof chunks were still
    // letting go -- and halving it settles the matter. Deep rather than wide:
    // bending capacity goes with the square of the depth.
    // One beam, across the depth only, and only where the house is deep enough
    // to need it.
    //
    // There were three, crossed, and they were the least reliable thing in the
    // city -- 9% of beam chunks shed under gravity alone however they were
    // sized. The honest fix was not a stiffer beam but a shorter span: a house
    // narrow enough that its roof reaches wall to wall needs no beam at all,
    // and a subsystem that is not there cannot fall down.
    const rMid = (rIn + rOut) / 2;
    if (rOut - rIn > 5) {
      put(rMid - 0.25, rMid + 0.25, fa, fb, top - 0.7, top, 'beam', 'wood-frame');
    }
    void rMid;

    // A wall plate: timber bedded along the wall head, which the roof lands on
    // instead of landing on the masonry.
    //
    // This is a MATERIAL fix, not a stiffness one. A bond takes the weaker of
    // the two materials it joins, so roof-on-wall resolved to white-stone --
    // and white-stone's tension limit is 0.8 MPa, because that is what masonry
    // actually is. Bonds resist moment, a roof bearing sags and rotates, and
    // that rotation put the joint in tension it had no business carrying:
    // roof-to-wall was the hottest class in the whole city at 2.9x, in tension
    // and compression at the same interface, which is a bending couple.
    //
    // Landing the roof on timber makes the same joint wood-to-wood at 14.4 MPa
    // and leaves the plate-to-masonry joint in pure bearing over the full wall
    // head, which is the one thing masonry is good at. It is also exactly what
    // a wall plate is for in a real building.
    const PLATE = 0.12;
    const plate = (rA, rB, from, to) =>
      put(rA, rB, from, to, top, top + PLATE, 'beam', 'wood-frame');
    plate(rIn, rOut, t0, t0 + dRad);
    plate(rIn, rOut, t1 - dRad, t1);
    plate(rOut - wallT, rOut, fa, fb);
    plate(rIn, rIn + wallT, fa, fb);
    // Thin. Now that the beams carry it in ~3 m panels the roof's own span is
    // trivial, and every extra centimetre is dead load on those beams: at
    // 0.4 m it weighed 240 kg/m^2 and was pulling them down with it.
    put(rIn, rOut, t0, t1, top + PLATE, top + PLATE + 0.22, 'roof', 'wood-frame');
  };

  for (let k = 1; k < C.tiers; k += 1) {
    const wallT = C.wallByTier[k - 1].t;
    // Outboard of the road, inboard of the wall. The road climbs to a full tier
    // above the terrace, so nothing is put in its sector.
    const rIn = r(k + 1) + C.roadWidth + 1.5;
    const rOut = r(k) - wallT - 0.5;
    if (rOut - rIn < 4) continue;
    const count = 14;
    const free = Math.PI * 2 - C.roadArc - 0.3;
    for (let i = 0; i < count; i += 1) {
      const angle = C.roadArc / 2 + 0.15 + (free * (i + 0.5)) / count;
      const arcWidth = Math.min((free / count) * 0.8, 4.2 / ((rIn + rOut) / 2));
      b.group(`house-t${k}-${i}`, () => house({ angle, arcWidth, rIn, rOut, base: y(k),
        height: k <= 3 ? 7 : 5, doorAt: 'inner' }));
    }
  }

  // ── the Citadel ──────────────────────────────────────────────────────────
  //
  // The Hall of Kings straddles the way from the Citadel gate to the prow, so
  // the last of the walk is through it: in at the south end, out at the north,
  // then east through the wall and onto the ledge.
  const cit = y(C.tiers);
  b.group('hall-of-kings', () => house({
    angle: -0.35, arcWidth: 1.15, rIn: 8, rOut: 20.5,
    base: cit, height: 12, doorAt: 'inner',
  }));

  // The nave colonnade. The Hall spans 12.5 m, which is a long way even in
  // timber; a row of piers down the middle halves it, and a hall of kings has
  // them anyway.
  for (let i = 0; i < 9; i += 1) {
    const a = -0.35 - 0.5 + (1.0 * (i + 0.5)) / 9;
    const dCol = 0.55 / 14;
    for (const rr of [[10.6, 11.7], [16.8, 17.9]]) {
      ringSegments({ rInner: rr[0], rOuter: rr[1], segments: 1, from: a - dCol, to: a + dCol })
        // Stopping under the beam rather than beside it: a column that runs
        // the full height shares space with the cross beam instead of
        // carrying it.
        .forEach((sg) => b.group('hall-of-kings', () => b.piece({ type: 'column',
          material: 'white-stone', axis: 'y', poly: sg.poly, lo: cit, hi: cit + 12 - 0.8 })));
    }
  }

  // The White Tower, on the far side of the Court of the Fountain. Solid: a
  // hollow shaft leaves its spire spanning the void at the top, which is the
  // same span failure as the roofs.
  for (let i = 0; i < 16; i += 1) {
    const t0 = (2 * Math.PI * i) / 16, t1 = (2 * Math.PI * (i + 1)) / 16;
    b.group('white-tower', () => b.piece({ type: 'wall', material: 'white-stone', axis: 'y',
      lo: cit, hi: cit + 34, poly: [[0, 0], ringPoint(0, 0, 5.0, t0), ringPoint(0, 0, 5.0, t1)] }));
  }
  for (let i = 0; i < 16; i += 1) {
    const t0 = (2 * Math.PI * i) / 16, t1 = (2 * Math.PI * (i + 1)) / 16;
    // The spire's base matches the shaft's radius exactly, so it is carried all
    // the way across rather than bridging a hole.
    b.group('white-tower', () => b.piece({ type: 'roof', material: 'white-stone', axis: 'y',
      lo: cit + 34, hi: cit + 42,
      poly: [[0, 0], ringPoint(0, 0, 5.0, t0), ringPoint(0, 0, 5.0, t1)] }));
  }

  // The prow — the pointed ledge over the plain, at Citadel level. Cantilevered
  // and pinned, so it needs nothing under it.
  b.piece({
    type: 'foundation', ...rock, axis: 'y', lo: y(C.tiers), hi: y(C.tiers) + 2,
    poly: [[r(C.tiers), -hw], [58, -2.5], [58, 2.5], [r(C.tiers), hw]],
  });

  b.build();
  return {
    pack: b.emit({ cameraTarget: [0, 40, 0], cameraDistance: 340 }),
    builder: b,
  };
}
