//! A single scene definition that any backend can run.
//!
//! This is the artifact behind "the same scene on both engines". The driver is
//! generic over [`PhysicsBackend`] and contains no engine-specific code, so
//! adding an engine adds zero lines here.
//!
//! It also encodes how the engines are *compared*. We do not expect PhysX and
//! Rapier to compute the same numbers and the assertions must not pretend
//! otherwise. Structural invariants (Tier 1) hold exactly on every engine;
//! conservation and settling (Tier 3) hold within per-engine bands; and the
//! cross-engine comparison (Tier 4) is calibrated against each engine's own
//! run-to-run spread rather than an invented tolerance.

use super::*;
use crate::types::Vec3;

/// A deterministic, engine-neutral scene: a lattice of boxes over a floor.
#[derive(Clone, Debug)]
pub struct SceneSpec {
    pub columns: u32,
    pub rows: u32,
    pub box_half: f32,
    pub spacing: f32,
    pub drop_height: f32,
    pub frames: u32,
    pub dt: f32,
}

impl Default for SceneSpec {
    fn default() -> Self {
        Self {
            columns: 4,
            rows: 5,
            box_half: 0.5,
            spacing: 1.02,
            drop_height: 0.6,
            frames: 180,
            dt: 1.0 / 60.0,
        }
    }
}

/// What a run produced. Deliberately coarse: these are the quantities that
/// mean the same thing on every engine.
#[derive(Clone, Debug, Default)]
pub struct SceneOutcome {
    pub bodies: usize,
    pub total_mass: f32,
    pub centroid: Vec3,
    pub max_abs_position: f32,
    pub sleeping_fraction: f32,
    pub peak_speed: f32,
    pub final_speed_sum: f32,
    pub any_non_finite: bool,
    /// Lowest y any dynamic body reached — catches tunnelling through the floor.
    pub min_y: f32,
}

/// Build and run the scene on `backend`.
pub fn run_scene<B: PhysicsBackend>(backend: &mut B, spec: &SceneSpec) -> SceneOutcome {
    let mut cmds: CommandBuffer<B::BodyId, B::ShapeId> = CommandBuffer::new();
    let mut out: CommandResults<B::BodyId, B::ShapeId> = CommandResults::default();

    // Floor: a fixed body with a wide, flat box.
    cmds.create_bodies.push(CreateBody {
        pose: Pose::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        kind: BodyKind::Fixed,
        linvel: Vec3::ZERO,
        angvel: Vec3::ZERO,
        ccd: false,
        start_sleeping: false,
    });
    let n = (spec.columns * spec.rows) as usize;
    for i in 0..n {
        let col = (i as u32 % spec.columns) as f32;
        let row = (i as u32 / spec.columns) as f32;
        cmds.create_bodies.push(CreateBody {
            pose: Pose::from_translation(Vec3::new(
                (col - spec.columns as f32 * 0.5) * spec.spacing,
                spec.drop_height + row * spec.spacing + spec.box_half,
                0.0,
            )),
            kind: BodyKind::Dynamic,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            ccd: false,
            start_sleeping: false,
        });
    }
    backend.apply(Phase::Topology, &cmds, &mut out).expect("create bodies");
    let all: Vec<B::BodyId> = out.created_bodies.clone();
    let floor = all[0];
    let dynamics: Vec<B::BodyId> = all[1..].to_vec();

    cmds.clear();
    cmds.create_shapes.push(CreateShape {
        body: floor,
        local: Pose::IDENTITY,
        geom: ShapeGeom::Cuboid { half_extents: Vec3::new(40.0, 1.0, 40.0) },
        mass: 0.0,
        node: 0,
    });
    for (i, b) in dynamics.iter().enumerate() {
        cmds.create_shapes.push(CreateShape {
            body: *b,
            local: Pose::IDENTITY,
            geom: ShapeGeom::Cuboid {
                half_extents: Vec3::new(spec.box_half, spec.box_half, spec.box_half),
            },
            mass: 1.0,
            node: (i + 1) as u32,
        });
    }
    backend.apply(Phase::Topology, &cmds, &mut out).expect("create shapes");

    cmds.clear();
    cmds.recompute_mass.extend(all.iter().copied());
    backend.apply(Phase::Topology, &cmds, &mut out).expect("recompute mass");

    // Settle the sleep thresholds so "at rest" means the same thing on both.
    cmds.clear();
    for b in &dynamics {
        cmds.set_sleep_thresholds.push((*b, 0.1, 0.1));
    }
    let _ = backend.apply(Phase::Tuning, &cmds, &mut out);

    let mut soa = BodyStateSoa::default();
    let mut contacts = ContactBatch::default();
    let mut result = SceneOutcome::default();
    result.min_y = f32::MAX;

    for _ in 0..spec.frames {
        backend.step(spec.dt);
        backend.drain_contacts(&mut contacts);
        backend.read_bodies(&dynamics, &mut soa);
        for i in 0..soa.len() {
            let p = soa.pose[i].translation;
            let s = soa.linvel[i].magnitude();
            if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() || !s.is_finite() {
                result.any_non_finite = true;
            }
            result.peak_speed = result.peak_speed.max(s);
            result.min_y = result.min_y.min(p.y);
        }
    }

    backend.read_bodies(&dynamics, &mut soa);
    let mut centroid = Vec3::ZERO;
    let mut mass = 0.0f32;
    let mut sleeping = 0usize;
    let mut speed_sum = 0.0f32;
    let mut max_abs = 0.0f32;
    for i in 0..soa.len() {
        let p = soa.pose[i].translation;
        centroid += p;
        mass += soa.mass[i];
        if soa.is_sleeping(i) {
            sleeping += 1;
        }
        speed_sum += soa.linvel[i].magnitude();
        max_abs = max_abs.max(p.x.abs()).max(p.y.abs()).max(p.z.abs());
    }
    let count = soa.len().max(1) as f32;
    result.bodies = soa.len();
    result.total_mass = mass;
    result.centroid = centroid / count;
    result.sleeping_fraction = sleeping as f32 / count;
    result.final_speed_sum = speed_sum;
    result.max_abs_position = max_abs;
    result
}

/// Tier 1 and 3 assertions: what must hold on *every* engine.
///
/// Returns the list of violations rather than panicking so a caller can report
/// all of them at once.
pub fn check_invariants(spec: &SceneSpec, o: &SceneOutcome) -> Vec<String> {
    let expected_bodies = (spec.columns * spec.rows) as usize;
    let mut v = Vec::new();

    if o.bodies != expected_bodies {
        v.push(format!("expected {expected_bodies} dynamic bodies, saw {}", o.bodies));
    }
    if o.any_non_finite {
        v.push("a pose or velocity went non-finite".into());
    }
    // Mass is authored: one unit per box, and nothing creates or destroys it.
    let want_mass = expected_bodies as f32;
    if (o.total_mass - want_mass).abs() > 0.01 * want_mass {
        v.push(format!("mass not conserved: {} vs authored {want_mass}", o.total_mass));
    }
    // Boxes dropped from a low height cannot leave the floor's extent.
    if o.max_abs_position > 60.0 {
        v.push(format!("a body escaped the scene: |p| reached {}", o.max_abs_position));
    }
    // Nothing may fall through the floor.
    if o.min_y < -2.0 {
        v.push(format!("a body tunnelled through the floor: min y = {}", o.min_y));
    }
    // Gravity cannot inject unbounded energy: a 0.6 m drop tops out well under
    // this, so exceeding it means the solver is adding energy.
    if o.peak_speed > 25.0 {
        v.push(format!("peak speed {} implies energy injection", o.peak_speed));
    }
    // The pile must come to rest. This is the assertion that catches a Tuning
    // pass which wakes bodies by rewriting unchanged values.
    if o.final_speed_sum > 0.5 * expected_bodies as f32 {
        v.push(format!("scene never settled: summed speed {}", o.final_speed_sum));
    }
    v
}

/// Tier 4: compare two engines against a band derived from their own
/// run-to-run spread, never against an invented constant.
pub fn cross_engine_within_band(
    a: &SceneOutcome,
    a_spread: f32,
    b: &SceneOutcome,
    b_spread: f32,
    slack: f32,
) -> Result<(), String> {
    let dy = (a.centroid.y - b.centroid.y).abs();
    // The floor is each engine's own reproducibility; anything tighter would
    // be measuring noise. A bit-reproducible engine contributes zero and the
    // test tightens automatically.
    let band = (a_spread.max(b_spread) * 4.0).max(slack);
    if dy > band {
        return Err(format!(
            "settled centroid differs by {dy:.4} m, outside the variance-calibrated band {band:.4} \
             (per-engine spreads {a_spread:.4} / {b_spread:.4})"
        ));
    }
    Ok(())
}
