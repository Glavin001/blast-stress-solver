use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::Mesh;
use bevy::render::render_resource::PrimitiveTopology;
use blast_stress_solver::rapier::{
    DebrisCleanupOptions, OptimizationMode, SmallBodyDampingOptions,
};
use blast_stress_solver::{
    ScenarioBond, ScenarioCollider, ScenarioDesc, ScenarioNode, Vec3 as SolverVec3,
};
use serde::Deserialize;

#[derive(Clone, Copy, Debug)]
pub enum EmbeddedSceneKey {
    FracturedWall,
    FracturedTower,
    FracturedBridge,
    BrickBuilding,
}

#[derive(Clone, Debug)]
pub struct SceneMeshAsset {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl SceneMeshAsset {
    pub fn to_bevy_mesh(&self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals.clone());
        mesh.insert_indices(Indices::U32(self.indices.clone()));
        mesh
    }
}

#[derive(Clone, Debug)]
pub struct LoadedScenePack {
    pub title: String,
    pub camera_target: bevy::prelude::Vec3,
    pub camera_distance: f32,
    pub projectile_radius: f32,
    pub projectile_mass: f32,
    pub projectile_speed: f32,
    pub projectile_ttl_secs: f32,
    pub gravity: f32,
    pub material_scale: f32,
    /// Explicit decoupled stress limits if the pack provided them.
    pub stress_limits: Option<StressLimits>,
    pub skip_single_bodies: bool,
    pub small_body_damping: SmallBodyDampingOptions,
    pub debris_cleanup: DebrisCleanupOptions,
    pub scenario: ScenarioDesc,
    pub node_meshes: Vec<SceneMeshAsset>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenePackJson {
    version: u32,
    title: String,
    defaults: SceneDefaultsJson,
    scenario: ScenarioJson,
    node_meshes: Vec<NodeMeshJson>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SceneDefaultsJson {
    camera: CameraDefaultsJson,
    projectile: ProjectileDefaultsJson,
    solver: SolverDefaultsJson,
    physics: PhysicsDefaultsJson,
    optimization: OptimizationDefaultsJson,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CameraDefaultsJson {
    target: Vec3Json,
    distance: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectileDefaultsJson {
    radius: f32,
    mass: f32,
    speed: f32,
    ttl_ms: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverDefaultsJson {
    gravity: f32,
    material_scale: f32,
    /// Optional explicit, decoupled stress limits (Pa). When present, the demo uses
    /// these verbatim instead of scaling base ratios by `material_scale`.
    #[serde(default)]
    limits: Option<LimitsJson>,
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

/// Explicit, decoupled stress limits (Pa) carried by a scene pack.
#[derive(Clone, Copy, Debug)]
pub struct StressLimits {
    pub compression_elastic: f32,
    pub compression_fatal: f32,
    pub tension_elastic: f32,
    pub tension_fatal: f32,
    pub shear_elastic: f32,
    pub shear_fatal: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PhysicsDefaultsJson {
    #[allow(dead_code)]
    friction: f32,
    #[allow(dead_code)]
    restitution: f32,
    #[allow(dead_code)]
    contact_force_scale: f32,
    #[serde(default)]
    skip_single_bodies: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OptimizationDefaultsJson {
    small_body_damping_mode: String,
    debris_cleanup_mode: String,
    debris_ttl_ms: f32,
    max_colliders_for_debris: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioJson {
    nodes: Vec<ScenarioNodeJson>,
    bonds: Vec<ScenarioBondJson>,
    node_sizes: Vec<Vec3Json>,
    node_colliders: Vec<NodeColliderJson>,
}

#[derive(Deserialize)]
struct ScenarioNodeJson {
    centroid: Vec3Json,
    mass: f32,
    volume: f32,
}

#[derive(Deserialize)]
struct ScenarioBondJson {
    node0: u32,
    node1: u32,
    centroid: Vec3Json,
    normal: Vec3Json,
    area: f32,
}

#[derive(Clone, Copy, Deserialize)]
struct Vec3Json {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum NodeColliderJson {
    Cuboid {
        #[serde(rename = "halfExtents")]
        half_extents: Vec3Json,
    },
    ConvexHull {
        points: Vec<f32>,
    },
}

#[derive(Deserialize)]
struct NodeMeshJson {
    positions: Vec<f32>,
    normals: Vec<f32>,
    indices: Vec<u32>,
}

pub fn load_embedded_scene_pack(key: EmbeddedSceneKey) -> Result<LoadedScenePack, String> {
    let payload = match key {
        EmbeddedSceneKey::FracturedWall => {
            include_str!("../assets/scenes/fractured-wall.json")
        }
        EmbeddedSceneKey::FracturedTower => {
            include_str!("../assets/scenes/fractured-tower.json")
        }
        EmbeddedSceneKey::FracturedBridge => {
            include_str!("../assets/scenes/fractured-bridge.json")
        }
        EmbeddedSceneKey::BrickBuilding => {
            include_str!("../assets/scenes/brick-building.json")
        }
    };
    parse_scene_pack(payload)
}

/// Load a scene pack from a JSON file at runtime. Used for generated, git-ignored
/// packs (e.g. high-rise.json) that are not embedded via `include_str!`.
pub fn load_scene_pack_file(path: &std::path::Path) -> Result<LoadedScenePack, String> {
    let payload = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "could not read scene pack {}: {error}\n\
             Generate it with: (cd ../blast-stress-solver && npm run build:ts && npm run generate:high-rise)",
            path.display()
        )
    })?;
    parse_scene_pack(&payload)
}

fn parse_scene_pack(payload: &str) -> Result<LoadedScenePack, String> {
    let pack: ScenePackJson = serde_json::from_str(payload)
        .map_err(|error| format!("invalid scene pack JSON: {error}"))?;
    if pack.version != 1 {
        return Err(format!("unsupported scene pack version {}", pack.version));
    }

    // `nodeMeshes` may be omitted (empty) for all-box scenes; in that case meshes
    // are derived from the per-node collider/size below. When present, counts must match.
    if !pack.node_meshes.is_empty() && pack.scenario.nodes.len() != pack.node_meshes.len() {
        return Err(format!(
            "scene pack node/mesh count mismatch: {} nodes vs {} meshes",
            pack.scenario.nodes.len(),
            pack.node_meshes.len()
        ));
    }
    if pack.scenario.nodes.len() != pack.scenario.node_sizes.len() {
        return Err(format!(
            "scene pack node/size count mismatch: {} nodes vs {} sizes",
            pack.scenario.nodes.len(),
            pack.scenario.node_sizes.len()
        ));
    }
    if pack.scenario.nodes.len() != pack.scenario.node_colliders.len() {
        return Err(format!(
            "scene pack node/collider count mismatch: {} nodes vs {} colliders",
            pack.scenario.nodes.len(),
            pack.scenario.node_colliders.len()
        ));
    }

    let node_meshes = if pack.node_meshes.is_empty() {
        // Derive a box mesh per node from its collider (cuboid) or size.
        pack.scenario
            .node_colliders
            .iter()
            .zip(pack.scenario.node_sizes.iter())
            .map(|(collider, size)| match collider {
                NodeColliderJson::Cuboid { half_extents } => {
                    box_mesh_asset(half_extents.x, half_extents.y, half_extents.z)
                }
                NodeColliderJson::ConvexHull { .. } => {
                    box_mesh_asset(size.x * 0.5, size.y * 0.5, size.z * 0.5)
                }
            })
            .collect::<Vec<_>>()
    } else {
        pack.node_meshes
            .iter()
            .map(parse_mesh_asset)
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(LoadedScenePack {
        title: pack.title,
        camera_target: bevy::prelude::Vec3::new(
            pack.defaults.camera.target.x,
            pack.defaults.camera.target.y,
            pack.defaults.camera.target.z,
        ),
        camera_distance: pack.defaults.camera.distance,
        projectile_radius: pack.defaults.projectile.radius,
        projectile_mass: pack.defaults.projectile.mass,
        projectile_speed: pack.defaults.projectile.speed,
        projectile_ttl_secs: pack.defaults.projectile.ttl_ms / 1000.0,
        gravity: pack.defaults.solver.gravity,
        material_scale: pack.defaults.solver.material_scale,
        stress_limits: pack.defaults.solver.limits.as_ref().map(|l| StressLimits {
            compression_elastic: l.compression_elastic,
            compression_fatal: l.compression_fatal,
            tension_elastic: l.tension_elastic,
            tension_fatal: l.tension_fatal,
            shear_elastic: l.shear_elastic,
            shear_fatal: l.shear_fatal,
        }),
        skip_single_bodies: pack.defaults.physics.skip_single_bodies,
        small_body_damping: SmallBodyDampingOptions {
            mode: parse_optimization_mode(&pack.defaults.optimization.small_body_damping_mode)?,
            collider_count_threshold: 3,
            min_linear_damping: 2.0,
            min_angular_damping: 2.0,
        },
        debris_cleanup: DebrisCleanupOptions {
            mode: parse_optimization_mode(&pack.defaults.optimization.debris_cleanup_mode)?,
            debris_ttl_secs: pack.defaults.optimization.debris_ttl_ms / 1000.0,
            max_colliders_for_debris: pack.defaults.optimization.max_colliders_for_debris,
        },
        scenario: ScenarioDesc {
            nodes: pack
                .scenario
                .nodes
                .iter()
                .map(|node| ScenarioNode {
                    centroid: node.centroid.into(),
                    mass: node.mass,
                    volume: node.volume,
                })
                .collect(),
            bonds: pack
                .scenario
                .bonds
                .iter()
                .map(|bond| ScenarioBond {
                    node0: bond.node0,
                    node1: bond.node1,
                    centroid: bond.centroid.into(),
                    normal: bond.normal.into(),
                    area: bond.area,
                })
                .collect(),
            node_sizes: pack
                .scenario
                .node_sizes
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            collider_shapes: pack
                .scenario
                .node_colliders
                .iter()
                .map(parse_node_collider)
                .collect::<Result<Vec<_>, _>>()?,
        },
        node_meshes,
    })
}

fn parse_mesh_asset(mesh: &NodeMeshJson) -> Result<SceneMeshAsset, String> {
    let positions = triples(&mesh.positions, "positions")?;
    let normals = triples(&mesh.normals, "normals")?;
    if positions.len() != normals.len() {
        return Err(format!(
            "mesh vertex/normal count mismatch: {} positions vs {} normals",
            positions.len(),
            normals.len()
        ));
    }
    Ok(SceneMeshAsset {
        positions,
        normals,
        indices: mesh.indices.clone(),
    })
}

/// Build a unit box mesh (centered at origin) with the given half-extents.
/// Used to derive node meshes for all-box scene packs that omit `nodeMeshes`.
fn box_mesh_asset(hx: f32, hy: f32, hz: f32) -> SceneMeshAsset {
    // 6 faces, each with 4 unique vertices (for flat per-face normals) and 2 tris.
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, -hz]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-hx, -hy, hz], [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]),
        // -Z
        ([0.0, 0.0, -1.0], [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]]),
    ];
    let mut positions = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, verts) in faces {
        let base = positions.len() as u32;
        for v in verts {
            positions.push(v);
            normals.push(normal);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    SceneMeshAsset {
        positions,
        normals,
        indices,
    }
}

fn triples(values: &[f32], label: &str) -> Result<Vec<[f32; 3]>, String> {
    if values.len() % 3 != 0 {
        return Err(format!(
            "{label} length must be divisible by 3, got {}",
            values.len()
        ));
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn parse_node_collider(collider: &NodeColliderJson) -> Result<Option<ScenarioCollider>, String> {
    match collider {
        NodeColliderJson::Cuboid { half_extents } => Ok(Some(ScenarioCollider::Cuboid {
            half_extents: (*half_extents).into(),
        })),
        NodeColliderJson::ConvexHull { points } => {
            let hull_points = triples(points, "convex hull points")?
                .into_iter()
                .map(|point| SolverVec3::new(point[0], point[1], point[2]))
                .collect();
            Ok(Some(ScenarioCollider::ConvexHull {
                points: hull_points,
            }))
        }
    }
}

fn parse_optimization_mode(value: &str) -> Result<OptimizationMode, String> {
    match value {
        "off" => Ok(OptimizationMode::Off),
        "always" => Ok(OptimizationMode::Always),
        "afterGroundCollision" => Ok(OptimizationMode::AfterGroundCollision),
        _ => Err(format!("unsupported optimization mode: {value}")),
    }
}

impl From<Vec3Json> for SolverVec3 {
    fn from(value: Vec3Json) -> Self {
        SolverVec3::new(value.x, value.y, value.z)
    }
}

#[cfg(test)]
mod tests {
    use super::{load_embedded_scene_pack, parse_scene_pack, EmbeddedSceneKey};

    /// A pack that omits `nodeMeshes` (all-box scene) and carries explicit limits —
    /// the shape produced by the high-rise generator. Meshes must be derived and the
    /// limits parsed.
    #[test]
    fn box_pack_without_meshes_derives_meshes_and_limits() {
        let json = r#"{
            "version": 1, "title": "t",
            "defaults": {
                "camera": {"target":{"x":0,"y":0,"z":0},"distance":10},
                "projectile": {"radius":0.5,"mass":1000,"speed":20,"ttlMs":8000},
                "solver": {"gravity":-9.81,"materialScale":1e10,
                    "limits":{"compressionElastic":12e6,"compressionFatal":30e6,
                              "tensionElastic":1.2e6,"tensionFatal":3e6,
                              "shearElastic":1.6e6,"shearFatal":4e6}},
                "physics": {"friction":0.25,"restitution":0,"contactForceScale":30},
                "optimization": {"smallBodyDampingMode":"always","debrisCleanupMode":"always",
                                 "debrisTtlMs":10000,"maxCollidersForDebris":3}
            },
            "scenario": {
                "nodes": [{"centroid":{"x":0,"y":0,"z":0},"mass":0,"volume":1},
                          {"centroid":{"x":0,"y":1,"z":0},"mass":100,"volume":1}],
                "bonds": [{"node0":0,"node1":1,"centroid":{"x":0,"y":0.5,"z":0},
                           "normal":{"x":0,"y":1,"z":0},"area":0.25}],
                "nodeSizes": [{"x":1,"y":1,"z":1},{"x":1,"y":1,"z":1}],
                "nodeColliders": [{"kind":"cuboid","halfExtents":{"x":0.5,"y":0.5,"z":0.5}},
                                  {"kind":"cuboid","halfExtents":{"x":0.5,"y":0.5,"z":0.5}}]
            },
            "nodeMeshes": []
        }"#;
        let pack = parse_scene_pack(json).expect("box pack should parse");
        assert_eq!(pack.node_meshes.len(), pack.scenario.nodes.len());
        assert!(pack.node_meshes[0].to_bevy_mesh().count_vertices() > 0);
        let limits = pack.stress_limits.expect("limits present");
        assert_eq!(limits.compression_fatal, 30e6);
        assert_eq!(limits.tension_fatal, 3e6);
    }

    #[test]
    fn fractured_wall_pack_loads() {
        let pack = load_embedded_scene_pack(EmbeddedSceneKey::FracturedWall)
            .expect("fractured wall pack should load");
        assert_eq!(pack.scenario.nodes.len(), pack.node_meshes.len());
        assert_eq!(
            pack.scenario.nodes.len(),
            pack.scenario.collider_shapes.len()
        );
        assert!(!pack.scenario.bonds.is_empty());
    }

    #[test]
    fn fractured_bridge_pack_contains_buildable_mesh() {
        let pack = load_embedded_scene_pack(EmbeddedSceneKey::FracturedBridge)
            .expect("fractured bridge pack should load");
        let mesh = pack.node_meshes[0].to_bevy_mesh();
        assert!(mesh.count_vertices() > 0);
    }

    #[test]
    fn brick_building_pack_loads() {
        let pack = load_embedded_scene_pack(EmbeddedSceneKey::BrickBuilding)
            .expect("brick building pack should load");
        assert_eq!(pack.scenario.nodes.len(), pack.node_meshes.len());
        assert_eq!(pack.scenario.nodes.len(), pack.scenario.node_sizes.len());
        assert_eq!(
            pack.scenario.nodes.len(),
            pack.scenario.collider_shapes.len()
        );
        assert!(!pack.scenario.bonds.is_empty());
    }
}
