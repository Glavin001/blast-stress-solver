//! The unified ScenePack loader against every shipped pack.
//!
//! Two defects in the loader this replaces motivate most of these assertions:
//! it computed limits as `BASE x materialScale`, so a v2 pack shipping
//! `materialScale: 1` received ~0.0009 Pa instead of ~48 MPa; and it discarded
//! every collider, so convex hulls silently became bounding boxes. Neither
//! failed loudly, which is why both survived.

#![cfg(feature = "scenarios")]

use std::path::PathBuf;

use blast_stress_solver::scene_pack::{
    load_scene_pack_file, parse_scene_pack, to_scenario_desc, SceneCollider, ScenePackError,
};

fn packs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in [
        "/root/workspace/vibe-land-3/destruction/assets/scenes",
        "/root/workspace/vibe-land-2/destruction/assets/scenes",
    ] {
        let dir = PathBuf::from(root);
        if !dir.is_dir() {
            continue;
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        found.sort();
        if !found.is_empty() {
            return found;
        }
        out = found;
    }
    out
}

#[test]
fn every_shipped_pack_loads() {
    let packs = packs();
    if packs.is_empty() {
        eprintln!("no scene packs present; skipping");
        return;
    }
    for path in &packs {
        let pack = load_scene_pack_file(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(!pack.nodes.is_empty(), "{}: no nodes", path.display());
        assert!(!pack.bonds.is_empty(), "{}: no bonds", path.display());
        assert!(!pack.materials.is_empty(), "{}: empty material table", path.display());
        assert!(
            pack.support_node_count() > 0,
            "{}: no support nodes, so nothing anchors it",
            path.display()
        );
        eprintln!(
            "[pack] {:<34} v{} nodes={:<6} bonds={:<6} materials={} types={} fallback={}",
            path.file_name().unwrap().to_string_lossy(),
            pack.version,
            pack.nodes.len(),
            pack.bonds.len(),
            pack.materials.len(),
            pack.node_types.len(),
            pack.used_fallback_limits
        );
    }
}

#[test]
fn v2_packs_carry_real_material_strength() {
    // The regression this exists for: limits synthesised from
    // `BASE x materialScale` gave a `materialScale: 1` pack ~0.0009 Pa.
    // Anything below 1 MPa here means the strength is being invented again.
    for path in packs() {
        let pack = load_scene_pack_file(&path).unwrap();
        if pack.version < 2 {
            continue;
        }
        for (i, m) in pack.materials.iter().enumerate() {
            assert!(
                m.limits.compression_elastic > 1.0e6,
                "{} material {i}: compression_elastic {} Pa is not a real material strength",
                path.display(),
                m.limits.compression_elastic
            );
            assert!(
                m.limits.compression_fatal >= m.limits.compression_elastic,
                "{} material {i}: fatal below elastic",
                path.display()
            );
        }
    }
}

#[test]
fn collider_geometry_survives_into_the_scenario() {
    // The other silent defect: `collider_shapes: Vec::new()` meant every convex
    // hull became an AABB, so shards collided and rendered as interpenetrating
    // boxes.
    let mut saw_hull = false;
    for path in packs() {
        let pack = load_scene_pack_file(&path).unwrap();
        let desc = to_scenario_desc(&pack);
        assert_eq!(
            desc.collider_shapes.len(),
            pack.nodes.len(),
            "{}: collider shapes dropped",
            path.display()
        );
        let hulls = pack
            .node_colliders
            .iter()
            .filter(|c| matches!(c, SceneCollider::ConvexHull { .. }))
            .count();
        if hulls > 0 {
            saw_hull = true;
            let carried = desc
                .collider_shapes
                .iter()
                .filter(|c| {
                    matches!(c, Some(blast_stress_solver::types::ScenarioCollider::ConvexHull { .. }))
                })
                .count();
            assert_eq!(
                hulls,
                carried,
                "{}: {hulls} hulls in the pack, {carried} reached the scenario",
                path.display()
            );
        }
    }
    assert!(saw_hull, "no pack exercised the convex-hull path");
}

#[test]
fn unknown_versions_are_rejected() {
    let json = r#"{"version":99,"scenario":{"nodes":[],"bonds":[]}}"#;
    assert!(matches!(
        parse_scene_pack(json),
        Err(ScenePackError::UnsupportedVersion(99))
    ));
}

#[test]
fn defaults_are_optional_all_the_way_down() {
    // The shipped district packs carry only `defaults.solver`, which is why
    // four of the five previous loaders could not open them at all.
    let json = r#"{
      "version":1,
      "scenario":{
        "nodes":[{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1},
                 {"centroid":{"x":0,"y":1,"z":0},"mass":1,"volume":1}],
        "bonds":[{"node0":0,"node1":1,"centroid":{"x":0,"y":0.5,"z":0},
                  "normal":{"x":0,"y":1,"z":0},"area":1}]
      }
    }"#;
    let pack = parse_scene_pack(json).expect("a pack with no defaults at all must load");
    assert!(pack.used_fallback_limits);
    assert_eq!(pack.materials.len(), 1);
}

#[test]
fn v2_without_materials_is_an_error_not_a_guess() {
    let json = r#"{
      "version":2,
      "defaults":{"solver":{"gravity":-9.81}},
      "scenario":{
        "nodes":[{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1}],
        "bonds":[]
      }
    }"#;
    assert!(matches!(parse_scene_pack(json), Err(ScenePackError::Invalid(_))));
}

#[test]
fn an_out_of_range_bond_material_names_the_bond() {
    let json = r#"{
      "version":2,
      "defaults":{"solver":{"materials":[
        {"compressionElastic":1e7,"compressionFatal":2e7,"tensionElastic":1e6,
         "tensionFatal":2e6,"shearElastic":1e6,"shearFatal":2e6}]}},
      "scenario":{
        "nodes":[{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1},
                 {"centroid":{"x":0,"y":1,"z":0},"mass":1,"volume":1}],
        "bonds":[{"node0":0,"node1":1,"centroid":{"x":0,"y":0.5,"z":0},
                  "normal":{"x":0,"y":1,"z":0},"area":1,"m":7}]
      }
    }"#;
    match parse_scene_pack(json) {
        Err(ScenePackError::Invalid(msg)) => {
            assert!(msg.contains("bond 0"), "error should name the bond: {msg}");
            assert!(msg.contains('7'), "error should name the index: {msg}");
        }
        other => panic!("expected an Invalid error, got {other:?}"),
    }
}

#[test]
fn a_bond_referencing_a_missing_node_is_rejected() {
    let json = r#"{
      "version":1,
      "scenario":{
        "nodes":[{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1}],
        "bonds":[{"node0":0,"node1":9,"centroid":{"x":0,"y":0,"z":0},
                  "normal":{"x":0,"y":1,"z":0},"area":1}]
      }
    }"#;
    assert!(matches!(parse_scene_pack(json), Err(ScenePackError::Invalid(_))));
}

#[test]
fn negative_limits_inherit_compression_at_parse() {
    // Resolved once, here, so no consumer ever sees a -1 sentinel and reads it
    // as a nonsensical strength.
    let json = r#"{
      "version":1,
      "defaults":{"solver":{"limits":{
        "compressionElastic":1e7,"compressionFatal":2e7,
        "tensionElastic":-1,"tensionFatal":-1,"shearElastic":-1,"shearFatal":-1}}},
      "scenario":{
        "nodes":[{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1}],
        "bonds":[]
      }
    }"#;
    let pack = parse_scene_pack(json).unwrap();
    let m = pack.materials[0].limits;
    assert_eq!(m.tension_elastic, 1e7);
    assert_eq!(m.tension_fatal, 2e7);
    assert_eq!(m.shear_elastic, 1e7);
    assert_eq!(m.shear_fatal, 2e7);
    assert!(!pack.used_fallback_limits, "explicit limits are not a fallback");
}
