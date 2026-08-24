//! ScenePack value types.
//!
//! `ScenePack` is deliberately **owned, cloneable and sliceable**: the building
//! tools (floor truncation, variants) construct new packs from existing ones,
//! which a borrowed view cannot support.

use crate::types::Vec3;

/// Per-material stress limits, in Pascals, fully resolved.
///
/// "Fully resolved" is the contract: a negative tension or shear limit in the
/// source JSON means *inherit compression*, and that sentinel is substituted at
/// parse time. A `-1.0` reaching a log, an assertion or a solver would read as a
/// nonsensical strength, so it never leaves the loader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressLimits {
    pub compression_elastic: f32,
    pub compression_fatal: f32,
    pub tension_elastic: f32,
    pub tension_fatal: f32,
    pub shear_elastic: f32,
    pub shear_fatal: f32,
}

/// Crush / comminution parameters (ScenePack v3).
///
/// `cap_pressure <= 0` disables crushing for the material, which is why it is
/// the field that decides whether the whole subsystem engages.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrushParams {
    pub cap_pressure: f32,
    pub cohesion: f32,
    pub friction_slope: f32,
    pub crush_energy: f32,
    pub crush_viscosity: f32,
    pub strain_rate_exponent: f32,
    pub reference_strain_rate: f32,
    pub debris_mass_fraction: f32,
    pub debris_fragment_count: u32,
}

impl CrushParams {
    pub fn enabled(&self) -> bool {
        self.cap_pressure > 0.0
    }
}

/// One material entry: limits plus optional crush.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneMaterial {
    pub limits: StressLimits,
    pub crush: Option<CrushParams>,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneNode {
    pub centroid: Vec3,
    pub mass: f32,
    pub volume: f32,
    /// Index into the material table (v3 `nodes[].m`). 0 when absent.
    pub material: u32,
}

impl SceneNode {
    /// Zero mass marks a node pinned to the world.
    ///
    /// There is no separate `supports` array in the format; support *is*
    /// zero mass.
    pub fn is_support(&self) -> bool {
        self.mass <= 0.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SceneBond {
    pub node0: u32,
    pub node1: u32,
    pub centroid: Vec3,
    /// The true contact-surface normal, not the centroid-to-centroid direction.
    ///
    /// This distinction is load-bearing: a shard bearing on a slab ledge has a
    /// diagonal centroid vector, so using it books the shard's own weight
    /// against the weak *shear* limit instead of concrete *compression*.
    pub normal: Vec3,
    pub area: f32,
    /// Index into the material table. 0 when absent.
    pub material: u32,
}

#[derive(Clone, Debug)]
pub enum SceneCollider {
    Cuboid { half_extents: Vec3 },
    ConvexHull { points: Vec<Vec3> },
}

/// A parsed ScenePack.
#[derive(Clone, Debug)]
pub struct ScenePack {
    pub version: u32,
    pub key: Option<String>,
    pub title: String,
    pub gravity: f32,
    /// Legacy TS scaling applied to the base limit ratios when a v1 pack
    /// authors no explicit limits.
    pub material_scale: f32,
    /// Always at least one entry. Index 0 is the structure default.
    pub materials: Vec<SceneMaterial>,
    pub nodes: Vec<SceneNode>,
    pub bonds: Vec<SceneBond>,
    /// Full extents (not half), parallel to `nodes`.
    pub node_sizes: Vec<Vec3>,
    /// Parallel to `nodes`. Empty only when the pack authored none.
    pub node_colliders: Vec<SceneCollider>,
    /// Structural role per node ("column", "slab", "wall", "infill",
    /// "foundation"). Parallel to `nodes`, or empty.
    ///
    /// **Any operation that reindexes nodes must remap this too** — floor
    /// truncation carrying the source list over would misattribute every role.
    pub node_types: Vec<String>,
    /// True when the limits came from the fallback rather than the pack.
    pub used_fallback_limits: bool,
}

impl ScenePack {
    pub fn support_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_support()).count()
    }

    /// Axis-aligned XZ footprint, from actual collider geometry.
    ///
    /// Hull vertices are unioned as world positions rather than treated as a
    /// symmetric half-extent: taking `max|p|` inflates the footprint for empty
    /// air on one side, which then inflates grid pitch for no reason.
    pub fn footprint_xz(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        let mut fold = |p: Vec3| {
            min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        };
        for (i, node) in self.nodes.iter().enumerate() {
            match self.node_colliders.get(i) {
                Some(SceneCollider::ConvexHull { points }) => {
                    for p in points {
                        fold(node.centroid + *p);
                    }
                }
                Some(SceneCollider::Cuboid { half_extents }) => {
                    fold(node.centroid - *half_extents);
                    fold(node.centroid + *half_extents);
                }
                None => {
                    let h = self.node_sizes.get(i).copied().unwrap_or(Vec3::ZERO) * 0.5;
                    fold(node.centroid - h);
                    fold(node.centroid + h);
                }
            }
        }
        if self.nodes.is_empty() {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (min, max)
        }
    }
}

#[derive(Debug)]
pub enum ScenePackError {
    Json(String),
    UnsupportedVersion(u32),
    CountMismatch(String),
    Invalid(String),
    Io(String),
}

impl std::fmt::Display for ScenePackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "scene pack json: {e}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported scene pack version {v}"),
            Self::CountMismatch(e) => write!(f, "scene pack count mismatch: {e}"),
            Self::Invalid(e) => write!(f, "invalid scene pack: {e}"),
            Self::Io(e) => write!(f, "scene pack io: {e}"),
        }
    }
}

impl std::error::Error for ScenePackError {}
