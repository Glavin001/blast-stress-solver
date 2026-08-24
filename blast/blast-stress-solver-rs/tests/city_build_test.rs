//! Building tools and multi-structure placement.
//!
//! The contracts here are inherited from implementations that existed twice —
//! C++ in `mini_city_main.cpp` and a Rust hand-port in vibe-land — so the
//! assertions pin the behaviour both were supposed to have.

#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use std::path::PathBuf;

use blast_stress_solver::backend::{PhysicsBackend, Pose};
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{DestructibleConfig, DestructibleSet, StructureId};
use blast_stress_solver::scene_pack::{
    building_offsets, load_scene_pack_file, make_building_variants, pitch_for_pack,
    solver_settings_for, to_scenario_desc, truncate_to_floors, variant_for_building,
};
use blast_stress_solver::types::Vec3;

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn pack_path(name: &str) -> Option<PathBuf> {
    for root in [
        "/root/workspace/vibe-land-3/destruction/assets/scenes",
        "/root/workspace/vibe-land-2/destruction/assets/scenes",
    ] {
        let p = PathBuf::from(root).join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[test]
fn truncation_remaps_every_parallel_array() {
    // The failure this guards: carrying `node_types` over instead of remapping
    // it misattributes every structural role, and because roles are metadata
    // nothing fails loudly — the building is simply authored wrong from then on.
    let Some(p) = pack_path("high-rise-3f-local.json") else {
        eprintln!("pack absent; skipping");
        return;
    };
    let full = load_scene_pack_file(&p).unwrap();
    assert!(!full.node_types.is_empty(), "this pack should carry nodeTypes");

    for floors in 1..=3 {
        let cut = truncate_to_floors(&full, floors, 3).expect("truncate");
        assert_eq!(cut.node_types.len(), cut.nodes.len(), "nodeTypes not remapped at {floors}f");
        assert_eq!(cut.node_sizes.len(), cut.nodes.len(), "nodeSizes not remapped at {floors}f");
        assert_eq!(
            cut.node_colliders.len(),
            cut.nodes.len(),
            "nodeColliders not remapped at {floors}f"
        );
        for b in &cut.bonds {
            assert!(
                (b.node0 as usize) < cut.nodes.len() && (b.node1 as usize) < cut.nodes.len(),
                "bond survived pointing at a dropped node"
            );
        }
        assert!(cut.support_node_count() > 0, "{floors}f left nothing anchored");
    }

    // Full height keeps everything: a ratio cutoff would drop the top row to
    // floating-point luck.
    let whole = truncate_to_floors(&full, 3, 3).unwrap();
    assert_eq!(whole.nodes.len(), full.nodes.len());
    assert_eq!(whole.bonds.len(), full.bonds.len());
}

#[test]
fn the_variant_ladder_matches_the_reference_contract() {
    // Pinned against the LIVE contract, not the C++ `--self-test` comment.
    //
    // `mini_city_main.cpp --self-test` still advertises 83/148/204 nodes and
    // 209/373/546 bonds. Those numbers are stale: on 2026-08-13 the shipped
    // `fractured-tower.json` had its centroids corrected to exact hull volume
    // centroids (they had been authored ~21 cm off, median). Truncation slices
    // by centroid Y, so a handful of nodes legitimately changed floors and the
    // 1- and 2-floor rungs moved. The full tower (204/546) is untouched, which
    // is exactly the signature of a cutoff shift rather than a parse bug.
    //
    // Verified three ways before re-pinning: recomputing the cutoff straight
    // from the asset JSON in f64 AND in f32 both yield 86/145/204, and
    // vibe-land's own `fractured_tower.rs::variant_counts_match_the_cpp_contract`
    // asserts (86, 213), (145, 365), (204, 546) and passes today.
    let Some(p) = pack_path("fractured-tower.json") else {
        eprintln!("fractured-tower absent; skipping");
        return;
    };
    let pack = load_scene_pack_file(&p).unwrap();
    let variants = make_building_variants(&pack, true);
    assert_eq!(variants.len(), 3, "expected a 1/2/3-floor ladder");

    let nodes: Vec<usize> = variants.iter().map(|v| v.pack.nodes.len()).collect();
    let bonds: Vec<usize> = variants.iter().map(|v| v.pack.bonds.len()).collect();
    let supports: Vec<usize> = variants.iter().map(|v| v.pack.support_node_count()).collect();
    eprintln!("[variants] nodes={nodes:?} bonds={bonds:?} supports={supports:?}");

    assert_eq!(nodes, vec![86, 145, 204], "node ladder drifted from the live contract");
    assert_eq!(bonds, vec![213, 365, 546], "bond ladder drifted from the live contract");
    assert_eq!(supports, vec![36, 36, 36], "support count must not change with height");
    assert!(variants[0].height < variants[2].height, "heights must increase");
}

#[test]
fn grid_pitch_leaves_a_street_between_buildings() {
    // The measured failure: an 18 m-wide high-rise at the reference demo's
    // hardcoded 18 m pitch has its facades touching. PhysX depenetrates them,
    // the weak infill shears on tick one, and the city demolishes itself before
    // anyone fires.
    let Some(p) = pack_path("high-rise-3f-local.json") else {
        eprintln!("pack absent; skipping");
        return;
    };
    let pack = load_scene_pack_file(&p).unwrap();
    let (min, max) = pack.footprint_xz();
    let width = (max.x - min.x).max(max.z - min.z);
    let clearance = 6.0;
    let pitch = pitch_for_pack(&pack, clearance);

    eprintln!("[pitch] footprint {width:.2} m -> pitch {pitch:.2} m");
    assert!(pitch >= width + clearance - 1e-3, "pitch must clear the true footprint");

    let offsets = building_offsets(3, pitch);
    assert_eq!(offsets.len(), 9);
    // Nearest neighbours must not overlap.
    for (i, a) in offsets.iter().enumerate() {
        for b in offsets.iter().skip(i + 1) {
            let d = ((a.x - b.x).powi(2) + (a.z - b.z).powi(2)).sqrt();
            assert!(d >= width, "buildings {d:.2} m apart with a {width:.2} m footprint");
        }
    }
}

#[test]
fn building_offsets_are_centred_and_row_major() {
    let o = building_offsets(2, 10.0);
    assert_eq!(o.len(), 4);
    assert_eq!(o[0], Vec3::new(-5.0, 0.0, -5.0));
    assert_eq!(o[1], Vec3::new(5.0, 0.0, -5.0));
    assert_eq!(o[3], Vec3::new(5.0, 0.0, 5.0));
    assert!(building_offsets(0, 10.0).is_empty(), "grid 0 must not underflow");
}

#[test]
fn variant_cycling_starts_tallest() {
    assert_eq!(variant_for_building(0, 3), 2);
    assert_eq!(variant_for_building(1, 3), 1);
    assert_eq!(variant_for_building(2, 3), 0);
    assert_eq!(variant_for_building(3, 3), 2);
    assert_eq!(variant_for_building(0, 0), 0, "no variants must not divide by zero");
}

/// Build a 2x2 city of variant towers and run it.
fn run_city<B: PhysicsBackend>(backend: &mut B) -> (usize, u32, usize) {
    let p = pack_path("high-rise-3f-local.json").expect("pack");
    let pack = load_scene_pack_file(&p).unwrap();
    let variants = make_building_variants(&pack, true);
    let pitch = pitch_for_pack(&pack, 6.0);
    let offsets = building_offsets(2, pitch);

    let mut set: DestructibleSet<B> = DestructibleSet::new();
    for (i, offset) in offsets.iter().enumerate() {
        let v = &variants[variant_for_building(i, variants.len())];
        let cfg = DestructibleConfig {
            world_pose: Pose::from_translation(*offset),
            gravity: G,
            solver: solver_settings_for(&v.pack, 0),
            ..Default::default()
        };
        set.attach(backend, StructureId(i as u32), &to_scenario_desc(&v.pack), cfg)
            .expect("attach");
    }

    for _ in 0..60 {
        backend.step(1.0 / 60.0);
        set.step(backend, 1.0 / 60.0);
    }
    (set.len(), set.actor_count(), set.body_count())
}

#[test]
fn a_multi_building_city_stands_on_both_engines() {
    if pack_path("high-rise-3f-local.json").is_none() {
        eprintln!("pack absent; skipping");
        return;
    }
    let mut rapier = RapierWorld::new(G);
    let r = run_city(&mut rapier);
    eprintln!("[rapier]    structures={} actors={} bodies={}", r.0, r.1, r.2);

    let mut physx = PhysXWorld::new_cpu(G, 2).expect("PhysX scene");
    let p = run_city(&mut physx);
    eprintln!("[physx-cpu] structures={} actors={} bodies={}", p.0, p.1, p.2);

    for (name, o) in [("rapier", r), ("physx-cpu", p)] {
        assert_eq!(o.0, 4, "[{name}] expected 4 buildings");
        // Each building stands: one actor apiece, and no fracture under gravity.
        assert_eq!(o.1, 4, "[{name}] a building fractured under its own weight");
        assert_eq!(o.2, o.1 as usize, "[{name}] body/actor bookkeeping diverged");
    }
    assert_eq!(r, p, "the same city produced different topology per engine");
}

#[test]
fn structures_are_independent_and_ids_are_unique() {
    let p = pack_path("high-rise-3f-local.json").expect("pack");
    let pack = load_scene_pack_file(&p).unwrap();
    let scenario = to_scenario_desc(&pack);
    let mut w = RapierWorld::new(G);
    let mut set: DestructibleSet<RapierWorld> = DestructibleSet::new();

    let cfg = || DestructibleConfig {
        gravity: G,
        solver: solver_settings_for(&pack, 0),
        ..Default::default()
    };
    set.attach(&mut w, StructureId(0), &scenario, cfg()).unwrap();
    assert!(
        set.attach(&mut w, StructureId(0), &scenario, cfg()).is_err(),
        "a duplicate structure id must be refused, not silently aliased"
    );
    set.attach(&mut w, StructureId(1), &scenario, cfg()).unwrap();
    assert_eq!(set.ids(), vec![StructureId(0), StructureId(1)]);
    assert!(set.get(StructureId(1)).is_some());
    assert!(set.get(StructureId(9)).is_none());
}
