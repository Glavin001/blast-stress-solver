//! Loader for the shared scene-pack JSON (the same files the Bevy demo embeds and
//! the web demo fetches). This lets headless Rust tests and parameter sweeps run on
//! the *exact* building that ships, with the *exact* tuned solver limits — keeping
//! the web and Rust behavior in lock-step from a single source of truth.
//!
//! Only the fields needed for the headless stress solver are parsed (nodes, bonds,
//! node sizes, and the solver defaults including the optional decoupled stress
//! `limits`). Meshes/colliders are ignored here.

use crate::{ScenarioBond, ScenarioDesc, ScenarioNode, SolverSettings, Vec3};
use serde::Deserialize;

// Base stress-limit ratios shared with the JS core (destructible-core.ts) and the
// Bevy demo (main.rs). Used only when a scene pack does not carry explicit limits.
const BASE_COMPRESSION_ELASTIC: f32 = 0.0009;
const BASE_COMPRESSION_FATAL: f32 = 0.0027;
const BASE_TENSION_ELASTIC: f32 = 0.0009;
const BASE_TENSION_FATAL: f32 = 0.0027;
const BASE_SHEAR_ELASTIC: f32 = 0.0012;
const BASE_SHEAR_FATAL: f32 = 0.0036;

#[derive(Clone, Copy, Deserialize)]
struct Vec3Json {
    x: f32,
    y: f32,
    z: f32,
}

impl From<Vec3Json> for Vec3 {
    fn from(v: Vec3Json) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

#[derive(Deserialize)]
struct NodeJson {
    centroid: Vec3Json,
    mass: f32,
    volume: f32,
}

#[derive(Deserialize)]
struct BondJson {
    node0: u32,
    node1: u32,
    centroid: Vec3Json,
    normal: Vec3Json,
    area: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioJson {
    nodes: Vec<NodeJson>,
    bonds: Vec<BondJson>,
    #[serde(default)]
    node_sizes: Vec<Vec3Json>,
    #[serde(default)]
    node_types: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LimitsJson {
    compression_elastic: f32,
    compression_fatal: f32,
    tension_elastic: f32,
    tension_fatal: f32,
    shear_elastic: f32,
    shear_fatal: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverJson {
    #[serde(default = "default_gravity")]
    gravity: f32,
    #[serde(default = "default_material_scale")]
    material_scale: f32,
    #[serde(default)]
    limits: Option<LimitsJson>,
}

fn default_gravity() -> f32 {
    -9.81
}
fn default_material_scale() -> f32 {
    1.0e10
}

#[derive(Deserialize)]
struct DefaultsJson {
    solver: SolverJson,
}

#[derive(Deserialize)]
struct PackJson {
    #[serde(default)]
    defaults: Option<DefaultsJson>,
    scenario: ScenarioJson,
}

/// A scene pack loaded for headless solving.
#[derive(Clone, Debug)]
pub struct LoadedScenario {
    pub scenario: ScenarioDesc,
    pub gravity: f32,
    pub material_scale: f32,
    /// Solver settings derived from the pack: explicit `limits` if present, otherwise
    /// the base ratios scaled by `material_scale` (matching the runtime demos).
    pub settings: SolverSettings,
    /// Per-node structural role (e.g. "column", "slab", "infill", "foundation").
    /// Empty if the pack did not carry `nodeTypes`.
    pub node_types: Vec<String>,
}

impl LoadedScenario {
    pub fn gravity_vec(&self) -> Vec3 {
        Vec3::new(0.0, self.gravity, 0.0)
    }

    /// True if the node at `index` is part of the load-bearing skeleton
    /// (column / beam / slab / floor / foundation), as opposed to infill.
    pub fn is_skeleton(&self, index: usize) -> bool {
        matches!(
            self.node_types.get(index).map(String::as_str),
            Some("column" | "beam" | "slab" | "floor" | "foundation")
        )
    }
}

fn settings_from(material_scale: f32, limits: &Option<LimitsJson>) -> SolverSettings {
    let mut s = SolverSettings {
        max_solver_iterations_per_frame: 24,
        ..SolverSettings::default()
    };
    match limits {
        Some(l) => {
            s.compression_elastic_limit = l.compression_elastic;
            s.compression_fatal_limit = l.compression_fatal;
            s.tension_elastic_limit = l.tension_elastic;
            s.tension_fatal_limit = l.tension_fatal;
            s.shear_elastic_limit = l.shear_elastic;
            s.shear_fatal_limit = l.shear_fatal;
        }
        None => {
            s.compression_elastic_limit = BASE_COMPRESSION_ELASTIC * material_scale;
            s.compression_fatal_limit = BASE_COMPRESSION_FATAL * material_scale;
            s.tension_elastic_limit = BASE_TENSION_ELASTIC * material_scale;
            s.tension_fatal_limit = BASE_TENSION_FATAL * material_scale;
            s.shear_elastic_limit = BASE_SHEAR_ELASTIC * material_scale;
            s.shear_fatal_limit = BASE_SHEAR_FATAL * material_scale;
        }
    }
    s
}

/// Parse a scene-pack JSON string into a [`LoadedScenario`].
pub fn load_scenario_str(json: &str) -> Result<LoadedScenario, String> {
    let pack: PackJson =
        serde_json::from_str(json).map_err(|e| format!("invalid scene pack JSON: {e}"))?;

    let nodes: Vec<ScenarioNode> = pack
        .scenario
        .nodes
        .iter()
        .map(|n| ScenarioNode {
            centroid: n.centroid.into(),
            mass: n.mass,
            volume: n.volume,
        })
        .collect();

    let bonds: Vec<ScenarioBond> = pack
        .scenario
        .bonds
        .iter()
        .map(|b| ScenarioBond {
            node0: b.node0,
            node1: b.node1,
            centroid: b.centroid.into(),
            normal: b.normal.into(),
            area: b.area,
        material: 0,
        })
        .collect();

    // Validate bond indices.
    for b in &bonds {
        if b.node0 as usize >= nodes.len() || b.node1 as usize >= nodes.len() {
            return Err(format!(
                "bond references out-of-range node ({}, {}) with {} nodes",
                b.node0,
                b.node1,
                nodes.len()
            ));
        }
    }

    let node_sizes: Vec<Vec3> = pack.scenario.node_sizes.iter().map(|v| (*v).into()).collect();

    let (gravity, material_scale, limits) = match pack.defaults {
        Some(d) => (d.solver.gravity, d.solver.material_scale, d.solver.limits),
        None => (default_gravity(), default_material_scale(), None),
    };

    let settings = settings_from(material_scale, &limits);

    Ok(LoadedScenario {
        scenario: ScenarioDesc {
            nodes,
            bonds,
            node_sizes,
            collider_shapes: Vec::new(),
        },
        gravity,
        material_scale,
        settings,
        node_types: pack.scenario.node_types,
    })
}

/// Load and parse a scene-pack JSON file from disk.
pub fn load_scenario_file(path: &std::path::Path) -> Result<LoadedScenario, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    load_scenario_str(&json)
}

/// Absolute path to the committed shared `high-rise.json` scene pack, resolved
/// relative to this crate (works in tests and examples regardless of CWD).
pub fn high_rise_scene_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../blast-stress-demo-rs/assets/scenes/high-rise.json")
}
