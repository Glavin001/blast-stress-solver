//! A/B: the old scenario loader against the unified ScenePack loader.
//!
//! The old path discarded collider geometry and synthesised stress limits.
//! This shows what that cost on real content.

use std::path::PathBuf;

use blast_stress_solver::scenarios::load_scenario_file;
use blast_stress_solver::scene_pack::{load_scene_pack_file, solver_settings_for, to_scenario_desc};
use blast_stress_solver::types::ScenarioCollider;

fn main() {
    let dir = PathBuf::from("/root/workspace/vibe-land-3/destruction/assets/scenes");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    files.sort();

    println!(
        "{:<34} {:>12} {:>12}   {:>7} {:>7}",
        "pack", "old c_elastic", "new c_elastic", "old hull", "new hull"
    );
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();

        let (old_limit, old_hulls) = match load_scenario_file(&f) {
            Ok(l) => {
                let hulls = l
                    .scenario
                    .collider_shapes
                    .iter()
                    .filter(|c| matches!(c, Some(ScenarioCollider::ConvexHull { .. })))
                    .count();
                (l.settings.compression_elastic_limit, hulls as i64)
            }
            Err(_) => (f32::NAN, -1),
        };

        let (new_limit, new_hulls) = match load_scene_pack_file(&f) {
            Ok(p) => {
                let d = to_scenario_desc(&p);
                let hulls = d
                    .collider_shapes
                    .iter()
                    .filter(|c| matches!(c, Some(ScenarioCollider::ConvexHull { .. })))
                    .count();
                (solver_settings_for(&p, 0).compression_elastic_limit, hulls as i64)
            }
            Err(_) => (f32::NAN, -1),
        };

        println!(
            "{name:<34} {old_limit:>12.4} {new_limit:>12.0}   {old_hulls:>7} {new_hulls:>7}"
        );
    }
}
