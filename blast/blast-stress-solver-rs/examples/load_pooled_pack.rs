//! Load a pack and report what its colliders resolved to.
//!
//! Exists because `kind: "shape"` is a reference into `shapeLibrary`, and a
//! parser that silently mis-resolved it would produce a structure whose nodes
//! collide as each other's shards -- visible only as physics behaving oddly.
//!
//!   cargo run --features scenarios --example load_pooled_pack -- <pack.json>
use blast_stress_solver::scene_pack::{parse_scene_pack, SceneCollider};

fn main() {
    let path = std::env::args().nth(1).expect("usage: load_pooled_pack <pack.json>");
    let text = std::fs::read_to_string(&path).expect("read pack");
    let pack = parse_scene_pack(&text).expect("parse pack");
    let mut cuboids = 0usize;
    let mut hulls = 0usize;
    let mut min_pts = usize::MAX;
    let mut max_pts = 0usize;
    for collider in &pack.node_colliders {
        match collider {
            SceneCollider::Cuboid { .. } => cuboids += 1,
            SceneCollider::ConvexHull { points } => {
                hulls += 1;
                min_pts = min_pts.min(points.len());
                max_pts = max_pts.max(points.len());
            }
        }
    }
    println!(
        "{}: nodes {} bonds {} | cuboids {} hulls {} | hull verts {}..{}",
        path,
        pack.nodes.len(),
        pack.bonds.len(),
        cuboids,
        hulls,
        if min_pts == usize::MAX { 0 } else { min_pts },
        max_pts
    );
}
