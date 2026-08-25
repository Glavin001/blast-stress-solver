//! ScenePack -> the pipeline's input types.
//!
//! This is where the old loader lost two things that mattered: it discarded
//! every `nodeCollider` (so convex hulls became their bounding boxes) and it
//! synthesised stress limits from `BASE x materialScale` (so a v2 pack shipping
//! `materialScale: 1` ran ten orders of magnitude weak). Both are carried here.

use super::types::{ScenePack, SceneCollider};
use crate::types::{ScenarioBond, ScenarioCollider, ScenarioDesc, ScenarioNode, SolverSettings, Vec3};

/// Build the pipeline's scenario description, preserving collider geometry.
pub fn to_scenario_desc(pack: &ScenePack) -> ScenarioDesc {
    let nodes: Vec<ScenarioNode> = pack
        .nodes
        .iter()
        .map(|n| ScenarioNode { centroid: n.centroid, mass: n.mass, volume: n.volume })
        .collect();

    let bonds: Vec<ScenarioBond> = pack
        .bonds
        .iter()
        .map(|b| ScenarioBond {
            node0: b.node0,
            node1: b.node1,
            centroid: b.centroid,
            normal: b.normal,
            area: b.area,
        material: b.material,
        })
        .collect();

    let node_sizes = if pack.node_sizes.is_empty() {
        // Derive from colliders rather than leaving the field empty, because a
        // downstream cuboid fallback keyed on an empty size vector produces
        // 1 m cubes for everything.
        pack.node_colliders
            .iter()
            .map(|c| match c {
                SceneCollider::Cuboid { half_extents } => *half_extents * 2.0,
                SceneCollider::ConvexHull { points } => {
                    let mut max = Vec3::ZERO;
                    for p in points {
                        max = Vec3::new(max.x.max(p.x.abs()), max.y.max(p.y.abs()), max.z.max(p.z.abs()));
                    }
                    max * 2.0
                }
            })
            .collect()
    } else {
        pack.node_sizes.clone()
    };

    let collider_shapes: Vec<Option<ScenarioCollider>> = pack
        .node_colliders
        .iter()
        .map(|c| {
            Some(match c {
                SceneCollider::Cuboid { half_extents } => {
                    ScenarioCollider::Cuboid { half_extents: *half_extents }
                }
                SceneCollider::ConvexHull { points } => {
                    ScenarioCollider::ConvexHull { points: points.clone() }
                }
            })
        })
        .collect();

    ScenarioDesc { nodes, bonds, node_sizes, collider_shapes }
}

/// Solver settings for one material entry.
///
/// The core currently exposes a single global strength, so a caller picks the
/// entry it wants (index 0 is the structure default). The full table lives on
/// the pack and is handed to backends that support per-bond materials — the
/// distinction matters, because collapsing a table to one material makes the
/// frame, its slabs and its cladding all fail at the same load.
pub fn solver_settings_for(pack: &ScenePack, material: usize) -> SolverSettings {
    let m = pack
        .materials
        .get(material)
        .unwrap_or(&pack.materials[0])
        .limits;
    SolverSettings {
        max_solver_iterations_per_frame: 25,
        graph_reduction_level: 0,
        compression_elastic_limit: m.compression_elastic,
        compression_fatal_limit: m.compression_fatal,
        tension_elastic_limit: m.tension_elastic,
        tension_fatal_limit: m.tension_fatal,
        shear_elastic_limit: m.shear_elastic,
        shear_fatal_limit: m.shear_fatal,
    }
}
