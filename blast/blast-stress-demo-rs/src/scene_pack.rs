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
    /// Equivalent to `materials[0].limits`; kept for existing callers.
    pub stress_limits: Option<StressLimits>,
    /// Material table, always >= 1 entry. A v1 pack synthesizes one entry from
    /// `solver.limits` so consumers see one shape regardless of pack version.
    /// See SCENE_PACK_FORMAT.md.
    pub materials: Vec<SceneMaterial>,
    /// Authored node roles ("foundation", "column", "slab", "infill", ...),
    /// parallel to `scenario.nodes`. Empty when the pack omits `nodeTypes`.
    pub node_types: Vec<String>,
    /// Per-bond index into `materials`, parallel to `scenario.bonds`.
    pub bond_materials: Vec<u32>,
    /// Pack schema version as loaded (1 or 2).
    pub version: u32,
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
    /// v2: the material table. Required when `version` is 2.
    #[serde(default)]
    materials: Option<Vec<MaterialJson>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterialJson {
    #[serde(default)]
    name: Option<String>,
    #[serde(flatten)]
    limits: LimitsJson,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct LimitsJson {
    compression_elastic: f32,
    compression_fatal: f32,
    tension_elastic: f32,
    tension_fatal: f32,
    shear_elastic: f32,
    shear_fatal: f32,
}

/// One entry of the pack's material table. `name` is author-defined and exists
/// for reports and debugging — there is no material enum and the library ships
/// no material library; see SCENE_PACK_FORMAT.md. Ductility is the width of the
/// (fatal - elastic) band, independent of raw strength.
#[derive(Clone, Debug)]
pub struct SceneMaterial {
    pub name: String,
    pub limits: StressLimits,
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
    /// Optional authored node roles, parallel to `nodes`.
    #[serde(default)]
    node_types: Vec<String>,
}

#[derive(Deserialize)]
struct ScenarioNodeJson {
    centroid: Vec3Json,
    mass: f32,
    volume: f32,
}

#[derive(Deserialize)]
struct ScenarioBondJson {
    /// Material index into `defaults.solver.materials`; absent means 0.
    #[serde(default)]
    m: u32,
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
    if pack.version != 1 && pack.version != 2 {
        return Err(format!(
            "unsupported scene pack version {} (see SCENE_PACK_FORMAT.md)",
            pack.version
        ));
    }

    // Material table. v2 requires it; v1 synthesizes a single entry so every
    // consumer sees one shape regardless of pack version.
    let to_limits = |l: &LimitsJson| StressLimits {
        compression_elastic: l.compression_elastic,
        compression_fatal: l.compression_fatal,
        tension_elastic: l.tension_elastic,
        tension_fatal: l.tension_fatal,
        shear_elastic: l.shear_elastic,
        shear_fatal: l.shear_fatal,
    };
    let materials: Vec<SceneMaterial> = if pack.version >= 2 {
        let table = pack
            .defaults
            .solver
            .materials
            .as_ref()
            .filter(|table| !table.is_empty())
            .ok_or_else(|| {
                "scene pack v2 requires a non-empty defaults.solver.materials array".to_string()
            })?;
        table
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let limits = to_limits(&m.limits);
                if !(limits.compression_elastic >= 0.0)
                    || !(limits.compression_fatal >= limits.compression_elastic)
                {
                    return Err(format!(
                        "material '{}' needs compressionFatal >= compressionElastic >= 0",
                        m.name.clone().unwrap_or_else(|| i.to_string())
                    ));
                }
                Ok(SceneMaterial {
                    name: m.name.clone().unwrap_or_else(|| format!("material{i}")),
                    limits,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    } else if let Some(limits) = pack.defaults.solver.limits.as_ref() {
        vec![SceneMaterial {
            name: "pack-limits".to_string(),
            limits: to_limits(limits),
        }]
    } else {
        // v1 without limits: the demo's material_scale-derived defaults apply.
        // Named "unstated" so a report can say so rather than implying the pack
        // authored a material.
        vec![SceneMaterial {
            name: "unstated".to_string(),
            limits: StressLimits {
                compression_elastic: 1.0e6,
                compression_fatal: 2.0e6,
                tension_elastic: 1.0e6,
                tension_fatal: 2.0e6,
                shear_elastic: 1.0e6,
                shear_fatal: 2.0e6,
            },
        }]
    };

    // Out of range is a hard error, never a clamp — a silent clamp to material 0
    // turns an authoring typo into a mysteriously strong joint.
    let bond_materials: Vec<u32> = pack
        .scenario
        .bonds
        .iter()
        .enumerate()
        .map(|(i, bond)| {
            if bond.m as usize >= materials.len() {
                return Err(format!(
                    "scene pack bond {} references material {} but the table has {} entries",
                    i,
                    bond.m,
                    materials.len()
                ));
            }
            Ok(bond.m)
        })
        .collect::<Result<Vec<_>, String>>()?;

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
        stress_limits: if pack.version >= 2 || pack.defaults.solver.limits.is_some() {
            Some(materials[0].limits)
        } else {
            None
        },
        materials,
        bond_materials,
        node_types: pack.scenario.node_types.clone(),
        version: pack.version,
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

    /// Cross-runtime ScenePack conformance — Rust side.
    ///
    /// The ScenePack JSON is the contract that lets the same structure run
    /// under TS/Rapier, Rust/Rapier and C++/PhysX so their APIs, behavior and
    /// performance can be compared without the structure being a variable.
    /// That only holds if all three loaders interpret the file identically.
    ///
    /// All three load the SAME fixture and assert the SAME digest, so a loader
    /// that drifts fails its own suite. The digest pins interpretation of the
    /// ASSET, not simulation results — Rapier and PhysX legitimately produce
    /// different trajectories from identical input.
    ///
    /// See SCENE_PACK_FORMAT.md and the siblings:
    ///   demos/blast-stress-demo/tests/scene_pack_conformance_test.cpp
    ///   blast/blast-stress-solver/src/tests/scenePack.conformance.test.ts
    mod conformance {
        use super::super::parse_scene_pack;
        use std::collections::BTreeMap;

        const FIXTURE: &str = include_str!(
            "../../blast-stress-solver/assets/conformance/structure-conformance-v2.json"
        );

        #[test]
        fn matches_the_golden_digest() {
            let pack = parse_scene_pack(FIXTURE).expect("fixture must load");

            assert_eq!(pack.version, 2, "version");
            assert_eq!(pack.scenario.nodes.len(), 7, "nodeCount");
            assert_eq!(pack.scenario.bonds.len(), 7, "bondCount");
            assert_eq!(pack.materials.len(), 3, "materialCount");
            assert_eq!(
                pack.materials.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
                vec!["reinforced-concrete", "concrete", "drywall-track"],
                "materialNames"
            );

            let support = pack.scenario.nodes.iter().filter(|n| n.mass == 0.0).count();
            assert_eq!(support, 2, "supportNodeCount");

            let total_mass: f32 = pack.scenario.nodes.iter().map(|n| n.mass).sum();
            assert!((total_mass - 6000.0).abs() < 1e-3, "totalMassKg: {total_mass}");

            let total_area: f32 = pack.scenario.bonds.iter().map(|b| b.area).sum();
            assert!((total_area - 1.46).abs() < 1e-5, "totalBondAreaM2: {total_area}");

            let mut per_material = vec![0usize; pack.materials.len()];
            for m in &pack.bond_materials {
                per_material[*m as usize] += 1;
            }
            assert_eq!(per_material, vec![2, 2, 3], "bondsPerMaterial");

            let mut per_class: BTreeMap<String, usize> = BTreeMap::new();
            for bond in &pack.scenario.bonds {
                let mut pair = [
                    pack.node_types[bond.node0 as usize].clone(),
                    pack.node_types[bond.node1 as usize].clone(),
                ];
                pair.sort();
                *per_class.entry(pair.join("~")).or_insert(0) += 1;
            }
            let expected: BTreeMap<String, usize> = [
                ("column~foundation", 2usize),
                ("column~infill", 2),
                ("column~slab", 2),
                ("infill~infill", 1),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
            assert_eq!(per_class, expected, "bondsPerJointClass");

            assert!((pack.gravity - (-9.81)).abs() < 1e-6, "gravity");
        }

        #[test]
        fn routes_material_indices_to_the_right_bonds() {
            let pack = parse_scene_pack(FIXTURE).expect("fixture must load");
            // Footings use the frame default; facade clips are drywall-track.
            assert_eq!(pack.materials[pack.bond_materials[0] as usize].name, "reinforced-concrete");
            assert_eq!(pack.materials[pack.bond_materials[4] as usize].name, "drywall-track");
        }

        #[test]
        fn omitted_material_index_means_zero() {
            // Bonds 0 and 1 carry no `m` field at all in the fixture.
            let pack = parse_scene_pack(FIXTURE).expect("fixture must load");
            assert_eq!(pack.bond_materials[0], 0);
            assert_eq!(pack.bond_materials[1], 0);
        }

        #[test]
        fn rejects_v2_without_materials() {
            let mut json: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
            json["defaults"]["solver"].as_object_mut().unwrap().remove("materials");
            let error = parse_scene_pack(&json.to_string()).unwrap_err();
            assert!(error.contains("materials"), "unexpected error: {error}");
        }

        #[test]
        fn rejects_out_of_range_bond_material_instead_of_clamping() {
            // A silent clamp to material 0 would turn an authoring typo into a
            // mysteriously strong joint — the bug class this format prevents.
            let mut json: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
            json["scenario"]["bonds"][2]["m"] = serde_json::json!(99);
            let error = parse_scene_pack(&json.to_string()).unwrap_err();
            assert!(error.contains("references material 99"), "unexpected error: {error}");
        }

        #[test]
        fn v1_packs_still_load_via_a_synthesized_table() {
            let mut json: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
            let concrete = json["defaults"]["solver"]["materials"][1].clone();
            json["version"] = serde_json::json!(1);
            json["defaults"]["solver"]["limits"] = concrete;
            json["defaults"]["solver"].as_object_mut().unwrap().remove("materials");
            for bond in json["scenario"]["bonds"].as_array_mut().unwrap() {
                bond.as_object_mut().unwrap().remove("m");
            }

            let pack = parse_scene_pack(&json.to_string()).expect("v1 must still load");
            assert_eq!(pack.version, 1);
            assert_eq!(pack.materials.len(), 1);
            assert!(pack.bond_materials.iter().all(|m| *m == 0));
            assert_eq!(pack.stress_limits.unwrap().compression_elastic, 12e6);
        }
    }

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
