//! Building tools: slice a pack into floor variants and place them in a city.
//!
//! These existed twice — as C++ in `mini_city_main.cpp` and as a hand-port in
//! vibe-land — which is a drift factory. One implementation, feature-gated
//! because not every consumer builds cities.

use super::types::{ScenePack, SceneBond, SceneCollider, SceneNode, ScenePackError};
use crate::types::Vec3;

/// The floor ladder the reference city builds.
pub const MAXIMUM_FLOORS: u32 = 3;

/// Keep only the lowest `floors` of `MAXIMUM_FLOORS`, rebuilding the graph.
///
/// Every array parallel to `nodes` must be remapped, not carried over —
/// `node_types` especially. Carrying the source list would misattribute every
/// structural role, and because roles are only metadata nothing would fail
/// loudly; the building would simply be authored wrong from then on.
pub fn truncate_to_floors(
    source: &ScenePack,
    floors: u32,
    maximum_floors: u32,
) -> Result<ScenePack, ScenePackError> {
    if floors == 0 || maximum_floors == 0 {
        return Err(ScenePackError::Invalid("floor count must be non-zero".into()));
    }
    if source.nodes.is_empty() {
        return Err(ScenePackError::Invalid("cannot truncate an empty pack".into()));
    }

    let min_y = source.nodes.iter().fold(f32::MAX, |a, n| a.min(n.centroid.y));
    let max_y = source.nodes.iter().fold(f32::MIN, |a, n| a.max(n.centroid.y));
    // At full height keep everything: a ratio cutoff would drop the topmost
    // node to floating-point luck.
    // Association order matches the reference (`(range * floors) / maximum`,
    // left to right). Measured on the reference tower it does not change the
    // ladder -- f32 and f64 both give 86/145/204 -- but keeping the same order
    // as the C++ and the vibe-land port means any future divergence is a real
    // one rather than a rounding artifact nobody can attribute.
    let cutoff = if floors >= maximum_floors {
        max_y + 1.0
    } else {
        min_y + (max_y - min_y) * floors as f32 / maximum_floors as f32
    };

    let mut remap = vec![u32::MAX; source.nodes.len()];
    let mut nodes = Vec::new();
    let mut node_sizes = Vec::new();
    let mut node_colliders: Vec<SceneCollider> = Vec::new();
    let mut node_types = Vec::new();

    for (i, node) in source.nodes.iter().enumerate() {
        if node.centroid.y > cutoff {
            continue;
        }
        remap[i] = nodes.len() as u32;
        nodes.push(*node);
        if let Some(s) = source.node_sizes.get(i) {
            node_sizes.push(*s);
        }
        if let Some(c) = source.node_colliders.get(i) {
            node_colliders.push(c.clone());
        }
        if let Some(t) = source.node_types.get(i) {
            node_types.push(t.clone());
        }
    }

    // A bond survives only if both endpoints did.
    let bonds: Vec<SceneBond> = source
        .bonds
        .iter()
        .filter_map(|b| {
            let n0 = remap[b.node0 as usize];
            let n1 = remap[b.node1 as usize];
            if n0 == u32::MAX || n1 == u32::MAX {
                return None;
            }
            Some(SceneBond { node0: n0, node1: n1, ..*b })
        })
        .collect();

    let truncated = ScenePack {
        nodes,
        bonds,
        node_sizes,
        node_colliders,
        node_types,
        ..source.clone()
    };

    if truncated.nodes.is_empty() {
        return Err(ScenePackError::Invalid("truncation removed every node".into()));
    }
    if truncated.bonds.is_empty() {
        return Err(ScenePackError::Invalid("truncation removed every bond".into()));
    }
    // A structure with no anchor is not a building, it is debris.
    if truncated.support_node_count() == 0 {
        return Err(ScenePackError::Invalid(
            "truncation removed every support node".into(),
        ));
    }
    Ok(truncated)
}

/// One building variant: a pack plus how tall it ended up.
#[derive(Clone, Debug)]
pub struct BuildingVariant {
    pub pack: ScenePack,
    pub floors: u32,
    /// Top of the tallest chunk, for camera framing and clearance checks.
    pub height: f32,
}

/// The 1..=MAXIMUM_FLOORS ladder, so a skyline is not N identical towers.
pub fn make_building_variants(source: &ScenePack, varied: bool) -> Vec<BuildingVariant> {
    let first = if varied { 1 } else { MAXIMUM_FLOORS };
    (first..=MAXIMUM_FLOORS)
        .filter_map(|floors| {
            let pack = truncate_to_floors(source, floors, MAXIMUM_FLOORS).ok()?;
            let height = pack
                .nodes
                .iter()
                .enumerate()
                .fold(0.0f32, |a, (i, n)| {
                    let half = pack.node_sizes.get(i).map(|s| s.y * 0.5).unwrap_or(0.0);
                    a.max(n.centroid.y + half)
                });
            Some(BuildingVariant { pack, floors, height })
        })
        .collect()
}

/// Grid pitch derived from the pack's own XZ footprint.
///
/// The reference demo hardcodes 18 m, which is correct only for the 8 m tower
/// it was written against. At 18 m an 18 m-wide high-rise has its facades
/// touching its neighbour's: PhysX depenetrates the contact, the weak infill
/// bonds shear on the first tick, and the city demolishes itself before anyone
/// fires a shot. `clearance` is added to the true footprint, never assumed.
pub fn pitch_for_pack(pack: &ScenePack, clearance_m: f32) -> f32 {
    let (min, max) = pack.footprint_xz();
    let width = (max.x - min.x).max(max.z - min.z);
    if !width.is_finite() || width <= 0.0 {
        return clearance_m.max(1.0);
    }
    width + clearance_m.max(0.0)
}

/// Row-major grid offsets, centred on the origin.
pub fn building_offsets(grid: u32, pitch: f32) -> Vec<Vec3> {
    if grid == 0 {
        return Vec::new();
    }
    let half = (grid.saturating_sub(1)) as f32 * pitch * 0.5;
    let mut out = Vec::with_capacity((grid * grid) as usize);
    for row in 0..grid {
        for col in 0..grid {
            out.push(Vec3::new(
                -half + col as f32 * pitch,
                0.0,
                -half + row as f32 * pitch,
            ));
        }
    }
    out
}

/// Which variant a given building index uses: tallest first, cycling down.
pub fn variant_for_building(building: usize, variants: usize) -> usize {
    if variants == 0 {
        return 0;
    }
    (variants - 1) - (building % variants)
}

/// Sanity check on a node's collider vs its authored size.
pub fn node_half_extents(pack: &ScenePack, i: usize) -> Vec3 {
    match pack.node_colliders.get(i) {
        Some(SceneCollider::Cuboid { half_extents }) => *half_extents,
        Some(SceneCollider::ConvexHull { points }) => {
            let mut m = Vec3::ZERO;
            for p in points {
                m = Vec3::new(m.x.max(p.x.abs()), m.y.max(p.y.abs()), m.z.max(p.z.abs()));
            }
            m
        }
        None => pack.node_sizes.get(i).copied().unwrap_or(Vec3::ZERO) * 0.5,
    }
}

/// Unused-import guard for `SceneNode` in doc examples.
#[allow(dead_code)]
fn _uses(_: &SceneNode) {}
