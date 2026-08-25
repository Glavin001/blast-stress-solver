//! ScenePack: the authored description of a destructible structure.
//!
//! One loader, shared by every engine. `SCENE_PACK_FORMAT.md` is normative —
//! "if a loader disagrees with this file, the loader is wrong" — and five
//! implementations of a normative format was a drift factory.

pub mod convert;
pub mod parse;
pub mod tools;
pub mod types;

pub use convert::{solver_settings_for, to_scenario_desc};
pub use parse::{load_scene_pack_file, parse_scene_pack, FALLBACK_LIMITS, MAX_VERSION, MIN_VERSION};
pub use tools::{
    building_offsets, make_building_variants, pitch_for_pack, truncate_to_floors,
    variant_for_building, BuildingVariant, MAXIMUM_FLOORS,
};
pub use types::{
    CrushParams, SceneBond, SceneCollider, SceneMaterial, SceneNode, ScenePack, ScenePackError,
    StressLimits,
};
