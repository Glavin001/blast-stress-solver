//! Flywheel overspeed burst test — spin a structure until it tears itself apart.
//!
//! Every other example loads structures with gravity or impacts. This one uses the
//! solver's third force path, `add_centrifugal_acceleration`: a steel flywheel
//! (hub + 8 spokes + 64-segment rim) is clamped by its hub in a spin rig and ramped
//! up in RPM until rotation alone bursts it — the classic turbine/flywheel
//! overspeed test, run entirely on the core API (no `scenarios`, no `rapier`).
//!
//!   cargo run --release --example flywheel_burst
//!
//! What it demonstrates:
//!
//! - `add_centrifugal_acceleration` as a per-actor, position-dependent load
//!   (a uniform field like gravity cannot burst a free-spinning ring; only the
//!   ω²r centrifugal field can)
//! - the full fracture loop: overstress detection → fracture commands → actor
//!   splits, with fragments classified as rim blocks, arcs, or spoke assemblies
//! - per-actor debris stress: shed fragments keep tumbling at the wheel's
//!   angular velocity and are driven about their *own* center of mass — the
//!   tumbling-debris path this API exists for (here fragments shed at exactly
//!   the speed they survived, so they correctly hold together in flight)
//! - `get_excess_forces` to recover the fling force a fragment carries at the
//!   moment it separates
//! - a physics cross-check: observed burst RPM versus a classical pin-jointed
//!   hoop-stress floor, plus a material sweep verifying burst RPM scales with
//!   the square root of material strength

use blast_stress_solver::*;
use std::collections::HashMap;
use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Wheel geometry and material
// ---------------------------------------------------------------------------

const N_RIM: usize = 64; // rim blocks around the circumference
const N_SPOKES: usize = 8; // each spoke lands on every 8th rim block
const SPOKE_SEGS: usize = 3; // segments per spoke
const RIM_RADIUS: f32 = 1.0; // m, rim centerline
const HUB_RADIUS: f32 = 0.15; // m, spokes start here
const DENSITY: f32 = 7800.0; // kg/m^3, steel

const RIM_CROSS: f32 = 0.10; // m, square rim bar -> hoop bond area 0.01 m^2
const SPOKE_AREA: f32 = 0.05; // m^2, spoke cross section (chunky on purpose)
const WELD_AREA: f32 = 0.012; // m^2, spoke-tip-to-rim weld

const FATAL_LIMIT: f32 = 60.0e6; // Pa, fatal stress for every mode
const ELASTIC_RATIO: f32 = 0.9; // elastic limit at 90% of fatal: a narrow
                                // fatigue band so failures stay near-predictable

// Spin rig schedule.
const RPM_STEP: f32 = 25.0; // RPM added per hold period
const HOLD_FRAMES: u32 = 6; // frames at each RPM so the CG solve settles
const MAX_RPM: f32 = 20_000.0;
const TUMBLE_FRAMES: u32 = 120; // how long shed debris keeps its spin (~2 s)

#[derive(Clone, Copy, PartialEq)]
enum NodeKind {
    Hub,
    Spoke { spoke: usize },
    Rim { slot: usize },
}

struct Wheel {
    nodes: Vec<NodeDesc>,
    bonds: Vec<BondDesc>,
    kinds: Vec<NodeKind>,
    rim_block_mass: f32,
}

/// Build the flywheel: one mass-0 hub node (the rig's chuck), `N_SPOKES` spokes
/// of `SPOKE_SEGS` segments, and an `N_RIM`-block rim closed into a hoop.
/// Spoke tips weld onto every `N_RIM / N_SPOKES`-th rim block.
fn build_wheel() -> Wheel {
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let mut kinds = Vec::new();

    // Hub: mass 0 marks it as a fixed support — the test rig holding the shaft.
    nodes.push(NodeDesc {
        centroid: Vec3::ZERO,
        mass: 0.0,
        volume: 0.0,
    });
    kinds.push(NodeKind::Hub);

    let radial = |angle: f32, r: f32| Vec3::new(r * angle.cos(), r * angle.sin(), 0.0);

    // Spokes: segments from the hub radius out to the rim's inner face.
    let spoke_outer = RIM_RADIUS - RIM_CROSS * 0.5;
    let seg_len = (spoke_outer - HUB_RADIUS) / SPOKE_SEGS as f32;
    let spoke_seg_mass = seg_len * SPOKE_AREA * DENSITY;
    let mut spoke_tip = Vec::with_capacity(N_SPOKES);
    for s in 0..N_SPOKES {
        let angle = s as f32 * TAU / N_SPOKES as f32;
        let dir = radial(angle, 1.0);
        let mut prev = 0u32; // hub
        for seg in 0..SPOKE_SEGS {
            let r = HUB_RADIUS + (seg as f32 + 0.5) * seg_len;
            let index = nodes.len() as u32;
            nodes.push(NodeDesc {
                centroid: radial(angle, r),
                mass: spoke_seg_mass,
                volume: seg_len * SPOKE_AREA,
            });
            kinds.push(NodeKind::Spoke { spoke: s });
            let joint_r = HUB_RADIUS + seg as f32 * seg_len;
            bonds.push(BondDesc {
                centroid: radial(angle, joint_r),
                normal: dir,
                area: SPOKE_AREA,
                node0: prev,
                node1: index,
            });
            prev = index;
        }
        spoke_tip.push(prev);
    }

    // Rim: a closed hoop of blocks. Slot s*(N_RIM/N_SPOKES) sits on spoke s.
    let arc_len = TAU * RIM_RADIUS / N_RIM as f32;
    let rim_block_mass = arc_len * RIM_CROSS * RIM_CROSS * DENSITY;
    let rim_node = |slot: usize| -> u32 { (1 + N_SPOKES * SPOKE_SEGS + slot) as u32 };
    for slot in 0..N_RIM {
        let angle = slot as f32 * TAU / N_RIM as f32;
        nodes.push(NodeDesc {
            centroid: radial(angle, RIM_RADIUS),
            mass: rim_block_mass,
            volume: arc_len * RIM_CROSS * RIM_CROSS,
        });
        kinds.push(NodeKind::Rim { slot });
    }
    for slot in 0..N_RIM {
        let next = (slot + 1) % N_RIM;
        let a = nodes[rim_node(slot) as usize].centroid;
        let b = nodes[rim_node(next) as usize].centroid;
        bonds.push(BondDesc {
            centroid: (a + b) * 0.5,
            normal: (b - a).normalize(),
            area: RIM_CROSS * RIM_CROSS,
            node0: rim_node(slot),
            node1: rim_node(next),
        });
    }

    // Welds: spoke tips onto their rim blocks.
    let slots_per_spoke = N_RIM / N_SPOKES;
    for (s, &tip) in spoke_tip.iter().enumerate() {
        let angle = s as f32 * TAU / N_SPOKES as f32;
        bonds.push(BondDesc {
            centroid: radial(angle, spoke_outer),
            normal: radial(angle, 1.0),
            area: WELD_AREA,
            node0: tip,
            node1: rim_node(s * slots_per_spoke),
        });
    }

    Wheel {
        nodes,
        bonds,
        kinds,
        rim_block_mass,
    }
}

fn solver_settings(strength_scale: f32) -> SolverSettings {
    let fatal = FATAL_LIMIT * strength_scale;
    SolverSettings {
        max_solver_iterations_per_frame: 128,
        graph_reduction_level: 0,
        compression_elastic_limit: fatal * ELASTIC_RATIO,
        compression_fatal_limit: fatal,
        tension_elastic_limit: fatal * ELASTIC_RATIO,
        tension_fatal_limit: fatal,
        shear_elastic_limit: fatal * ELASTIC_RATIO,
        shear_fatal_limit: fatal,
    }
}

/// Classical pin-jointed (truss) burst estimate: a free hoop of point masses m
/// at radius R needs hoop force T = m ω² R / (2 sin(π/N)) per joint, so the
/// joints reach `fatal` stress at ω = sqrt(fatal·A·2sin(π/N)/(mR)).
///
/// This is a *floor*, not the expected burst speed. Blast bonds are rigid-body
/// welds, not pin joints: they also carry the radial load in bending (angular
/// impulses), so the wheel runs well past this estimate before the joints at
/// the spoke shoulders finally crack. What the solver must reproduce exactly
/// is the ω² scaling — see the strength sweep at the end.
fn predicted_burst_rpm(rim_block_mass: f32, strength_scale: f32) -> f32 {
    let hoop_area = RIM_CROSS * RIM_CROSS;
    let geometry = 2.0 * (std::f32::consts::PI / N_RIM as f32).sin();
    let omega = (FATAL_LIMIT * strength_scale * hoop_area * geometry
        / (rim_block_mass * RIM_RADIUS))
        .sqrt();
    omega * 60.0 / TAU
}

// ---------------------------------------------------------------------------
// Fragment bookkeeping
// ---------------------------------------------------------------------------

fn center_of_mass(wheel: &Wheel, node_ids: &[u32]) -> (Vec3, f32) {
    let mut com = Vec3::ZERO;
    let mut mass = 0.0f32;
    for &n in node_ids {
        let node = &wheel.nodes[n as usize];
        com += node.centroid * node.mass;
        mass += node.mass;
    }
    if mass > 0.0 {
        (com / mass, mass)
    } else {
        (Vec3::ZERO, 0.0)
    }
}

fn describe_fragment(wheel: &Wheel, node_ids: &[u32]) -> String {
    let rim = node_ids
        .iter()
        .filter(|&&n| matches!(wheel.kinds[n as usize], NodeKind::Rim { .. }))
        .count();
    let spoke = node_ids
        .iter()
        .filter(|&&n| matches!(wheel.kinds[n as usize], NodeKind::Spoke { .. }))
        .count();
    let blocks = |n: usize| if n == 1 { "block" } else { "blocks" };
    match (rim, spoke) {
        (1, 0) => "rim block".to_string(),
        (r, 0) => format!("rim arc ({r} blocks)"),
        (0, s) => format!("spoke section ({s} segs)"),
        (r, s) => format!("spoke assembly ({s} segs + {r} rim {})", blocks(r)),
    }
}

/// One line of wheel state: every rim slot and spoke, attached or gone.
fn wheel_art(wheel: &Wheel, hub_nodes: &[u32]) -> String {
    let mut rim = vec!['·'; N_RIM];
    let mut spokes = vec!['·'; N_SPOKES];
    for &n in hub_nodes {
        match wheel.kinds[n as usize] {
            NodeKind::Rim { slot } => rim[slot] = '█',
            NodeKind::Spoke { spoke } => spokes[spoke] = '█',
            NodeKind::Hub => {}
        }
    }
    format!(
        "rim [{}]  spokes [{}]",
        rim.into_iter().collect::<String>(),
        spokes.into_iter().collect::<String>()
    )
}

// ---------------------------------------------------------------------------
// The burst test
// ---------------------------------------------------------------------------

fn main() {
    let wheel = build_wheel();
    println!("FLYWHEEL OVERSPEED BURST TEST");
    println!(
        "  wheel: {} nodes, {} bonds  (hub + {} spokes x {} segs + {}-block rim, R = {} m)",
        wheel.nodes.len(),
        wheel.bonds.len(),
        N_SPOKES,
        SPOKE_SEGS,
        N_RIM,
        RIM_RADIUS,
    );
    let predicted = predicted_burst_rpm(wheel.rim_block_mass, 1.0);
    println!(
        "  material: fatal stress {:.0} MPa -> pin-jointed truss floor: {predicted:.0} RPM \
         (rigid bonds also carry bending, so expect several times that)",
        FATAL_LIMIT / 1.0e6,
    );
    println!("  schedule: +{RPM_STEP} RPM every {HOLD_FRAMES} frames, redline {MAX_RPM} RPM\n");

    let mut solver = ExtStressSolver::new(&wheel.nodes, &wheel.bonds, &solver_settings(1.0))
        .expect("failed to create solver");

    let mut rpm = 0.0f32;
    let mut frame = 0u32;
    let mut first_overstress_rpm: Option<f32> = None;
    let mut first_burst_rpm: Option<f32> = None;
    let mut first_spoke_loss_rpm: Option<f32> = None;
    let mut secondary_fractures = 0usize;
    let mut shed_fragments = 0usize;
    // Debris still spinning at shed time: actor index -> (own COM, frames left).
    let mut tumbling: HashMap<u32, (Vec3, u32)> = HashMap::new();

    loop {
        frame += 1;
        if frame.is_multiple_of(HOLD_FRAMES) {
            rpm += RPM_STEP;
        }
        if rpm > MAX_RPM {
            println!("\nreached redline {MAX_RPM} RPM with the remaining structure intact");
            break;
        }
        let omega = Vec3::new(0.0, 0.0, rpm * TAU / 60.0);

        // Drive the wheel: the actor still holding the hub spins about the shaft.
        let actors = solver.actors();
        let hub_actor = actors
            .iter()
            .find(|a| a.nodes.contains(&0))
            .expect("hub actor must exist");
        let hub_node_count = hub_actor.nodes.len();
        solver.add_centrifugal_acceleration(hub_actor.actor_index, Vec3::ZERO, omega);

        // Drive shed debris about its own center of mass — fragments leave the
        // wheel still rotating at ω, and big arcs can shatter again in flight.
        tumbling.retain(|&actor, (com, frames_left)| {
            *frames_left -= 1;
            *frames_left > 0 && solver.add_centrifugal_acceleration(actor, *com, omega)
        });

        solver.update();

        if first_overstress_rpm.is_none() && solver.overstressed_bond_count() > 0 {
            first_overstress_rpm = Some(rpm);
            println!("{rpm:>5.0} RPM  first overstressed bonds (fatigue band entered)");
        }

        let commands = solver.generate_fracture_commands();
        if commands.is_empty() {
            continue;
        }
        let broken: usize = commands.iter().map(|c| c.bond_fractures.len()).sum();
        if first_burst_rpm.is_none() {
            first_burst_rpm = Some(rpm);
            println!("{rpm:>5.0} RPM  the rim is cracking ({broken} bonds taking damage)");
        }
        let hub_actor_index = hub_actor.actor_index;
        secondary_fractures += commands
            .iter()
            .filter(|c| c.actor_index != hub_actor_index)
            .map(|c| c.bond_fractures.len())
            .sum::<usize>();

        let events = solver.apply_fracture_commands(&commands);
        // (description, fling force in N) — forces are averaged when the
        // wheel's symmetry sheds several identical fragments at once.
        let mut shed_lines: Vec<(String, Option<f32>)> = Vec::new();
        for event in &events {
            let from_debris = event.parent_actor_index != hub_actor_index;
            let parent_tumble = tumbling.remove(&event.parent_actor_index);
            for child in &event.children {
                if child.nodes.contains(&0) {
                    continue; // still the wheel
                }
                let (com, mass) = center_of_mass(&wheel, &child.nodes);
                let what = describe_fragment(&wheel, &child.nodes);
                if from_debris {
                    // A tumbling fragment broke up further: children keep spinning.
                    if let Some((_, frames_left)) = parent_tumble {
                        if child.nodes.len() > 1 {
                            tumbling.insert(child.actor_index, (com, frames_left));
                        }
                    }
                    shed_lines.push((format!("mid-air breakup -> {what}"), None));
                    continue;
                }
                shed_fragments += 1;
                if child.nodes.len() > 1 {
                    tumbling.insert(child.actor_index, (com, TUMBLE_FRAMES));
                }
                let tip_speed = omega.z * (com.x * com.x + com.y * com.y).sqrt();
                let fling = solver
                    .get_excess_forces(child.actor_index, com)
                    .map(|(f, _)| f.magnitude())
                    .unwrap_or(0.0);
                shed_lines.push((
                    format!(
                        "shed {what}: {mass:.0} kg leaving at {tip_speed:.0} m/s ({:.0} km/h)",
                        tip_speed * 3.6,
                    ),
                    Some(fling),
                ));
                if first_spoke_loss_rpm.is_none() && what.contains("spoke") {
                    first_spoke_loss_rpm = Some(rpm);
                }
            }
        }

        // Re-read the wheel for the status line; only narrate frames that shed.
        let actors = solver.actors();
        let hub_actor = actors
            .iter()
            .find(|a| a.nodes.contains(&0))
            .expect("hub actor must exist");
        if !shed_lines.is_empty() {
            println!(
                "{rpm:>5.0} RPM  {}  {broken} bonds broke",
                wheel_art(&wheel, &hub_actor.nodes)
            );
            // The wheel is symmetric, so fragments come off in identical sets.
            let mut grouped: Vec<(String, usize, f32)> = Vec::new();
            for (line, fling) in shed_lines {
                match grouped.iter_mut().find(|(l, _, _)| *l == line) {
                    Some((_, n, force)) => {
                        *n += 1;
                        *force += fling.unwrap_or(0.0);
                    }
                    None => grouped.push((line, 1, fling.unwrap_or(0.0))),
                }
            }
            for (line, n, force) in grouped {
                let prefix = if n == 1 {
                    String::new()
                } else {
                    format!("{n} x ")
                };
                if force > 0.0 {
                    println!(
                        "           - {prefix}{line}, excess force ~{:.1} MN",
                        force / n as f32 / 1.0e6
                    );
                } else {
                    println!("           - {prefix}{line}");
                }
            }
        }

        if hub_actor.nodes.len() == 1 && hub_node_count > 1 {
            println!("\nthe wheel is gone — only the bare hub is left in the chuck");
            break;
        }
    }

    println!("\nSUMMARY");
    println!("  pin-jointed truss floor   : {predicted:.0} RPM");
    if let Some(r) = first_overstress_rpm {
        println!("  first overstress observed : {r:.0} RPM");
    }
    if let Some(r) = first_burst_rpm {
        println!("  first fracture observed   : {r:.0} RPM");
    }
    if let Some(r) = first_spoke_loss_rpm {
        println!("  first spoke torn off      : {r:.0} RPM");
    }
    println!("  fragments shed            : {shed_fragments}");
    println!("  mid-air (secondary) breaks: {secondary_fractures} bonds");
    println!("  final actor count         : {}", solver.actor_count());

    // -----------------------------------------------------------------------
    // Scaling-law sweep: hoop stress grows with ω², so burst RPM must scale
    // with sqrt(material strength). Materials at 1x / 4x / 9x strength should
    // burst at RPM ratios of 1 : 2 : 3.
    // -----------------------------------------------------------------------
    println!("\nSCALING CHECK  (burst RPM should scale with sqrt of material strength)");
    println!("  strength   observed burst   ratio to baseline");
    let mut baseline: Option<f32> = None;
    for scale in [1.0f32, 4.0, 9.0] {
        match burst_rpm_quiet(&wheel, scale) {
            Some(observed) => {
                let base = *baseline.get_or_insert(observed);
                println!(
                    "  {scale:>4.0}x       {observed:>7.0} RPM     {:.2}x (sqrt predicts {:.2}x)",
                    observed / base,
                    scale.sqrt(),
                );
            }
            None => println!("  {scale:>4.0}x       no burst within the ramp"),
        }
    }
}

/// Ramp a fresh wheel at `strength_scale` material strength and return the RPM
/// of the first fracture, without printing the play-by-play.
fn burst_rpm_quiet(wheel: &Wheel, strength_scale: f32) -> Option<f32> {
    let mut solver =
        ExtStressSolver::new(&wheel.nodes, &wheel.bonds, &solver_settings(strength_scale))
            .expect("failed to create solver");
    let mut rpm = 0.0f32;
    let mut frame = 0u32;
    loop {
        frame += 1;
        if frame.is_multiple_of(HOLD_FRAMES) {
            rpm += RPM_STEP;
        }
        if rpm > MAX_RPM * strength_scale.sqrt() {
            return None;
        }
        let omega = Vec3::new(0.0, 0.0, rpm * TAU / 60.0);
        let actors = solver.actors();
        let hub_actor = actors.iter().find(|a| a.nodes.contains(&0))?;
        solver.add_centrifugal_acceleration(hub_actor.actor_index, Vec3::ZERO, omega);
        solver.update();
        if !solver.generate_fracture_commands().is_empty() {
            return Some(rpm);
        }
    }
}
