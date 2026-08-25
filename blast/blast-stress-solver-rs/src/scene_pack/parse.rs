//! The one ScenePack parser.
//!
//! Replaces five divergent loaders. Two properties of the previous
//! implementations drove the design:
//!
//! **Every `defaults` sub-object is optional, including `solver` itself.** The
//! shipped district and downtown packs carry only `defaults.solver`; four of the
//! five old loaders required `camera` and `projectile` and therefore could not
//! open them at all.
//!
//! **Nothing is silently substituted.** The old library loader computed limits
//! as `BASE × materialScale`, so a v2 pack shipping `materialScale: 1` with no
//! `limits` block received ~0.0009 Pa instead of ~48 MPa — ten orders of
//! magnitude weak, with no warning. Here, a v2 pack must carry materials, and a
//! v1 pack falling back says so.

use serde::Deserialize;

use super::types::*;
use crate::types::Vec3;

/// Limits used when a **v1** pack authors none: reference concrete.
///
/// The blast demo loaders used a 1/2 MPa placeholder, so the same pack loaded
/// ~12× weaker there than in the game. This value is what production runs, and
/// it is now the single answer. Using it sets `used_fallback_limits`, which the
/// loader reports rather than hiding.
pub const FALLBACK_LIMITS: StressLimits = StressLimits {
    compression_elastic: 12.0e6,
    compression_fatal: 30.0e6,
    tension_elastic: 1.2e6,
    tension_fatal: 3.0e6,
    shear_elastic: 1.6e6,
    shear_fatal: 4.0e6,
};

pub const MIN_VERSION: u32 = 1;
pub const MAX_VERSION: u32 = 3;

#[derive(Deserialize)]
struct Vec3Json {
    x: f32,
    y: f32,
    z: f32,
}
impl From<&Vec3Json> for Vec3 {
    fn from(v: &Vec3Json) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PackJson {
    version: u32,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    defaults: Option<DefaultsJson>,
    scenario: ScenarioJson,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DefaultsJson {
    #[serde(default)]
    solver: Option<SolverJson>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverJson {
    #[serde(default)]
    gravity: Option<f32>,
    #[serde(default)]
    material_scale: Option<f32>,
    #[serde(default)]
    limits: Option<LimitsJson>,
    #[serde(default)]
    materials: Option<Vec<MaterialJson>>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct LimitsJson {
    compression_elastic: f32,
    compression_fatal: f32,
    tension_elastic: f32,
    tension_fatal: f32,
    shear_elastic: f32,
    shear_fatal: f32,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MaterialJson {
    #[serde(flatten)]
    limits: LimitsJson,
    #[serde(default)]
    crush: Option<CrushJson>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CrushJson {
    #[serde(default)]
    cap_pressure: f32,
    #[serde(default)]
    cohesion: f32,
    #[serde(default)]
    friction_slope: f32,
    #[serde(default)]
    crush_energy: f32,
    #[serde(default)]
    crush_viscosity: f32,
    #[serde(default)]
    strain_rate_exponent: f32,
    #[serde(default)]
    reference_strain_rate: f32,
    #[serde(default)]
    debris_mass_fraction: f32,
    #[serde(default)]
    debris_fragment_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioJson {
    nodes: Vec<NodeJson>,
    bonds: Vec<BondJson>,
    #[serde(default)]
    node_sizes: Vec<Vec3Json>,
    #[serde(default)]
    node_colliders: Vec<Option<ColliderJson>>,
    #[serde(default)]
    node_types: Vec<String>,
}

#[derive(Deserialize)]
struct NodeJson {
    centroid: Vec3Json,
    mass: f32,
    volume: f32,
    #[serde(default)]
    m: Option<u32>,
}

#[derive(Deserialize)]
struct BondJson {
    node0: u32,
    node1: u32,
    centroid: Vec3Json,
    normal: Vec3Json,
    area: f32,
    #[serde(default)]
    m: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ColliderJson {
    kind: String,
    #[serde(default)]
    half_extents: Option<Vec3Json>,
    #[serde(default)]
    points: Vec<f32>,
}

/// A negative tension or shear limit means "inherit compression".
fn resolve_inherited(l: &LimitsJson) -> StressLimits {
    let inherit = |v: f32, fallback: f32| if v < 0.0 { fallback } else { v };
    StressLimits {
        compression_elastic: l.compression_elastic,
        compression_fatal: l.compression_fatal,
        tension_elastic: inherit(l.tension_elastic, l.compression_elastic),
        tension_fatal: inherit(l.tension_fatal, l.compression_fatal),
        shear_elastic: inherit(l.shear_elastic, l.compression_elastic),
        shear_fatal: inherit(l.shear_fatal, l.compression_fatal),
    }
}

fn validate_material(i: usize, l: &StressLimits) -> Result<(), ScenePackError> {
    let bad = |what: &str| {
        Err(ScenePackError::Invalid(format!("material {i}: {what}")))
    };
    if l.compression_elastic < 0.0 {
        return bad("compressionElastic is negative");
    }
    if l.compression_fatal < l.compression_elastic {
        return bad("compressionFatal is below compressionElastic");
    }
    if l.tension_fatal < l.tension_elastic {
        return bad("tensionFatal is below tensionElastic");
    }
    if l.shear_fatal < l.shear_elastic {
        return bad("shearFatal is below shearElastic");
    }
    Ok(())
}

fn crush_from(c: &CrushJson) -> CrushParams {
    CrushParams {
        cap_pressure: c.cap_pressure,
        cohesion: c.cohesion,
        friction_slope: c.friction_slope,
        crush_energy: c.crush_energy,
        crush_viscosity: c.crush_viscosity,
        strain_rate_exponent: c.strain_rate_exponent,
        reference_strain_rate: c.reference_strain_rate,
        debris_mass_fraction: c.debris_mass_fraction,
        debris_fragment_count: c.debris_fragment_count,
    }
}

/// Exact-duplicate point removal.
///
/// Authored buffers repeat corners per face and the repeats are byte-identical,
/// so this is lossless — the hull is untouched. **No geometric thinning.** An
/// earlier implementation strided every Nth point to fit the GPU's 64-vertex
/// cap and silently deformed colliders away from the rendered geometry; the cap
/// belongs to the cooker, which computes the optimal bounded hull.
fn dedup_points(flat: &[f32]) -> Vec<Vec3> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(flat.len() / 3);
    for p in flat.chunks_exact(3) {
        if seen.insert([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]) {
            out.push(Vec3::new(p[0], p[1], p[2]));
        }
    }
    out
}

pub fn parse_scene_pack(payload: &str) -> Result<ScenePack, ScenePackError> {
    let raw: PackJson =
        serde_json::from_str(payload).map_err(|e| ScenePackError::Json(e.to_string()))?;

    if raw.version < MIN_VERSION || raw.version > MAX_VERSION {
        return Err(ScenePackError::UnsupportedVersion(raw.version));
    }

    let solver = raw.defaults.as_ref().and_then(|d| d.solver.as_ref());
    let gravity = solver.and_then(|s| s.gravity).unwrap_or(-9.81);
    let material_scale = solver.and_then(|s| s.material_scale).unwrap_or(1.0);

    // Materials. v2+ must author a table; v1 may carry a single `limits` block,
    // and otherwise falls back — visibly.
    let mut used_fallback = false;
    let materials: Vec<SceneMaterial> = match solver.and_then(|s| s.materials.as_ref()) {
        Some(list) => {
            if list.is_empty() {
                return Err(ScenePackError::Invalid(
                    "materials table is present but empty".into(),
                ));
            }
            list.iter()
                .enumerate()
                .map(|(i, m)| {
                    let limits = resolve_inherited(&m.limits);
                    validate_material(i, &limits)?;
                    Ok(SceneMaterial { limits, crush: m.crush.as_ref().map(crush_from) })
                })
                .collect::<Result<_, ScenePackError>>()?
        }
        None => {
            if raw.version >= 2 {
                return Err(ScenePackError::Invalid(format!(
                    "version {} requires defaults.solver.materials",
                    raw.version
                )));
            }
            let limits = match solver.and_then(|s| s.limits.as_ref()) {
                Some(l) => resolve_inherited(l),
                None => {
                    used_fallback = true;
                    FALLBACK_LIMITS
                }
            };
            validate_material(0, &limits)?;
            vec![SceneMaterial { limits, crush: None }]
        }
    };

    let scenario = raw.scenario;
    let node_count = scenario.nodes.len();
    if node_count == 0 {
        return Err(ScenePackError::Invalid("scenario has no nodes".into()));
    }
    if !scenario.node_sizes.is_empty() && scenario.node_sizes.len() != node_count {
        return Err(ScenePackError::CountMismatch(format!(
            "nodeSizes {} vs nodes {node_count}",
            scenario.node_sizes.len()
        )));
    }
    if !scenario.node_colliders.is_empty() && scenario.node_colliders.len() != node_count {
        return Err(ScenePackError::CountMismatch(format!(
            "nodeColliders {} vs nodes {node_count}",
            scenario.node_colliders.len()
        )));
    }
    if !scenario.node_types.is_empty() && scenario.node_types.len() != node_count {
        return Err(ScenePackError::CountMismatch(format!(
            "nodeTypes {} vs nodes {node_count}",
            scenario.node_types.len()
        )));
    }

    let nodes: Vec<SceneNode> = scenario
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let material = n.m.unwrap_or(0);
            if material as usize >= materials.len() {
                return Err(ScenePackError::Invalid(format!(
                    "node {i} names material {material}, table has {}",
                    materials.len()
                )));
            }
            Ok(SceneNode {
                centroid: (&n.centroid).into(),
                mass: n.mass,
                // A zero or missing volume would divide by zero in the crush
                // strain rate, which normalises by the chunk's characteristic
                // size.
                volume: if n.volume > 0.0 { n.volume } else { 1.0 },
                material,
            })
        })
        .collect::<Result<_, _>>()?;

    let bonds: Vec<SceneBond> = scenario
        .bonds
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if b.node0 as usize >= node_count || b.node1 as usize >= node_count {
                return Err(ScenePackError::Invalid(format!(
                    "bond {i} references node {}/{} of {node_count}",
                    b.node0, b.node1
                )));
            }
            let material = b.m.unwrap_or(0);
            // Never clamp: an out-of-range index is an authoring error, and
            // clamping it silently reassigns the material.
            if material as usize >= materials.len() {
                return Err(ScenePackError::Invalid(format!(
                    "bond {i} names material {material}, table has {}",
                    materials.len()
                )));
            }
            Ok(SceneBond {
                node0: b.node0,
                node1: b.node1,
                centroid: (&b.centroid).into(),
                normal: (&b.normal).into(),
                area: if b.area > 0.0 { b.area } else { 1.0 },
                material,
            })
        })
        .collect::<Result<_, _>>()?;

    let node_sizes: Vec<Vec3> = scenario.node_sizes.iter().map(Vec3::from).collect();

    let node_colliders: Vec<SceneCollider> = scenario
        .node_colliders
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let Some(c) = c else {
                let h = node_sizes.get(i).copied().unwrap_or(Vec3::new(1.0, 1.0, 1.0)) * 0.5;
                return Ok(SceneCollider::Cuboid { half_extents: h });
            };
            match c.kind.as_str() {
                "cuboid" => {
                    let h = c
                        .half_extents
                        .as_ref()
                        .map(Vec3::from)
                        .or_else(|| node_sizes.get(i).map(|s| *s * 0.5))
                        .ok_or_else(|| {
                            ScenePackError::Invalid(format!("node {i}: cuboid without extents"))
                        })?;
                    Ok(SceneCollider::Cuboid { half_extents: h })
                }
                "convex_hull" | "convexHull" => {
                    let points = dedup_points(&c.points);
                    if points.len() < 4 {
                        return Err(ScenePackError::Invalid(format!(
                            "node {i}: convex hull has {} distinct points",
                            points.len()
                        )));
                    }
                    Ok(SceneCollider::ConvexHull { points })
                }
                other => Err(ScenePackError::Invalid(format!(
                    "node {i}: unknown collider kind {other:?}"
                ))),
            }
        })
        .collect::<Result<_, _>>()?;

    Ok(ScenePack {
        version: raw.version,
        key: raw.key,
        title: raw.title.unwrap_or_default(),
        gravity,
        material_scale,
        materials,
        nodes,
        bonds,
        node_sizes,
        node_colliders,
        node_types: scenario.node_types,
        used_fallback_limits: used_fallback,
    })
}

pub fn load_scene_pack_file(path: &std::path::Path) -> Result<ScenePack, ScenePackError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ScenePackError::Io(format!("{}: {e}", path.display())))?;
    let pack = parse_scene_pack(&text)?;
    if pack.used_fallback_limits {
        // Visible, not silent: a v1 pack with no authored strength is running
        // on a substituted material, and which substitute you get used to
        // differ by 12x between runtimes.
        eprintln!(
            "[scene_pack] {}: v1 pack authors no limits; using the reference-concrete fallback \
             ({:.0} MPa compressive)",
            path.display(),
            FALLBACK_LIMITS.compression_elastic / 1.0e6
        );
    }
    Ok(pack)
}
