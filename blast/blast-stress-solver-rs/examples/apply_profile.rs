//! Apply-path micro-profiler — isolates the cost of the **Rapier topology edits** a split
//! triggers (body create / recycle / retire, collider re-parent / insert), with **no stress
//! solve, no physics step, and no resim**. We build a structure once, then drive
//! `Sim::apply_split_only` with a contrived event and read the per-op `SplitEditStats`.
//!
//! Two regimes, to exercise the cheap and the expensive Rapier calls:
//!   * **shatter (1×N)** — one body lets go into N fragments. Create/insert-dominated:
//!     N-1 `bodies.insert` + collider re-parents. (The common forward-fracture case.)
//!   * **merge (M→M/2)** — N singleton bodies regroup into N/2 two-node children, each
//!     pulling nodes from two different bodies. Reuse + **retire**-dominated: the planner
//!     reuses one body per child and retires the other, so this hits `bodies.remove`
//!     (the costliest op: islands + broad-phase + contacts + joints).
//!
//! Run: `cargo run --release --example apply_profile --features bench-support`

use blast_stress_solver::bench_harness::*;
use blast_stress_solver::rapier::SplitEditStats;

fn print_stats(name: &str, s: &SplitEditStats) {
    println!("\n── {name} ──");
    println!(
        "  counts    reuse={:<5} recycle={:<5} create={:<5} retire={:<5}",
        s.reused_bodies, s.recycled_bodies, s.created_bodies, s.retired_bodies
    );
    println!(
        "            colliders: moved={:<5} inserted={:<5} removed={:<5}  body_type_flips={}",
        s.moved_colliders, s.inserted_colliders, s.removed_colliders, s.body_type_flips
    );
    println!("  plan      {:.3} ms   (assignment only — separate from apply)", s.plan_ms);
    println!("  APPLY     {:.3} ms   total topology edits", s.apply_ms);
    let mut rows = [
        ("body_create  (insert)", s.body_create_ms),
        ("body_retire  (remove)", s.body_retire_ms),
        ("collider_move(set_parent)", s.collider_move_ms),
        ("collider_insert", s.collider_insert_ms),
        ("child_pose", s.child_pose_ms),
        ("velocity_fit (recompute_mass+COM)", s.velocity_fit_ms),
        ("sleep_init", s.sleep_init_ms),
    ];
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (label, ms) in rows {
        let pct = if s.apply_ms > 0.0 { 100.0 * ms / s.apply_ms } else { 0.0 };
        println!("              {label:<34} {ms:>8.3} ms  ({pct:>4.0}%)");
    }
}

fn main() {
    println!("Apply-path micro-profiler — topology edits only (no solve / step / resim)");

    // Square single-layer walls: node count = side². Each starts as one connected body.
    for side in [8u32, 16, 32, 48] {
        let scn = wall(side, side, 1);
        let w = scn.nodes.len() as u32;

        // 1×N shatter on the fresh (single-body) structure.
        let mut sim = Sim::new(&scn, SimConfig::default());
        let r = sim.apply_split_only(&shatter_event(w));
        print_stats(&format!("wall {side}×{side}  —  shatter 1×{w}"), &r.stats);

        // M→M/2 merge: shatter first (setup, not measured) so we have w singleton bodies,
        // then regroup adjacent pairs — each child straddles two bodies.
        let mut sim = Sim::new(&scn, SimConfig::default());
        let _setup = sim.apply_split_only(&shatter_event(w));
        let r = sim.apply_split_only(&merge_pairs_event(w / 2));
        print_stats(&format!("wall {side}×{side}  —  merge {w}→{} (reparent)", w / 2), &r.stats);
    }
}
