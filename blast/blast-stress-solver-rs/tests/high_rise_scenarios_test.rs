//! Headless behavior tests for the shared high-rise apartment scene pack.
//!
//! These verify the "realistic, non-glass" destruction bands using the *exact*
//! committed scene pack and its tuned, decoupled concrete limits:
//!   (a) stable under gravity (no fractures, single actor, base intact);
//!   (b) a weak wrecking ball blows out local INFILL only (skeleton untouched);
//!   (c) a strong ball at a column punches a BOUNDED hole (not a glass cascade);
//!   (d) deterministic (identical bond-break counts across runs).
//!
//! NOTE: ExtStressSolver is a *stress* solver — it breaks bonds and splits actors
//! but does not integrate falling/topple (that is Rapier's job in the full demos).
//! So we assert on bonds-broken (total + by type) and actor count, which is the
//! glass-vs-local signal. COM/topple is covered by the JS Rapier test + the sweep.
//!
//! The scene pack is a generated, git-ignored artifact. If it is missing the tests
//! SKIP (so a fresh `cargo test` without a Node build still passes); CI generates it
//! first so the assertions actually run. Regenerate with:
//!   (cd ../blast-stress-solver && npm run build && npm run generate:high-rise)

#[cfg(feature = "scenarios")]
mod high_rise {
    use blast_stress_solver::scenarios::*;
    use blast_stress_solver::*;

    #[derive(Default, Debug)]
    struct Metrics {
        total_fractures: u32,
        skeleton_fractures: u32,
        infill_fractures: u32,
        foundation_fractures: u32,
        peak_actor_count: u32,
        final_actor_count: u32,
    }

    fn load() -> Option<LoadedScenario> {
        match load_scenario_file(&high_rise_scene_path()) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("SKIP high-rise test (scene pack not generated): {e}");
                None
            }
        }
    }

    /// Step the solver `frames` times, optionally injecting a per-frame impact force
    /// at `impact` (node, position, force) for the first `impact_frames` frames.
    /// Returns fracture metrics classified by node type.
    fn run(
        loaded: &LoadedScenario,
        frames: u32,
        impact: Option<(u32, Vec3, Vec3)>,
        impact_frames: u32,
    ) -> Metrics {
        let (nodes, bonds) = loaded.scenario.to_solver_descs();
        let mut solver = ExtStressSolver::new(&nodes, &bonds, &loaded.settings).unwrap();
        let g = loaded.gravity_vec();
        let mut m = Metrics::default();

        for frame in 0..frames {
            solver.add_gravity(g);
            if let Some((node, pos, force)) = impact {
                if frame < impact_frames {
                    solver.add_force(node, pos, force, ForceMode::Force);
                }
            }
            solver.update();
            if solver.overstressed_bond_count() > 0 {
                let cmds = solver.generate_fracture_commands();
                for c in &cmds {
                    for bf in &c.bond_fractures {
                        m.total_fractures += 1;
                        let a = loaded.is_skeleton(bf.node_index0 as usize);
                        let b = loaded.is_skeleton(bf.node_index1 as usize);
                        let is_foundation = is_foundation(loaded, bf.node_index0)
                            || is_foundation(loaded, bf.node_index1);
                        if is_foundation {
                            m.foundation_fractures += 1;
                        }
                        if a && b {
                            m.skeleton_fractures += 1;
                        } else if !a && !b {
                            m.infill_fractures += 1;
                        } else {
                            // mixed skeleton<->infill joint: count as infill detachment
                            m.infill_fractures += 1;
                        }
                    }
                }
                if !cmds.is_empty() {
                    solver.apply_fracture_commands(&cmds);
                }
            }
            m.peak_actor_count = m.peak_actor_count.max(solver.actor_count());
        }
        m.final_actor_count = solver.actor_count();
        m
    }

    fn is_foundation(loaded: &LoadedScenario, node: u32) -> bool {
        loaded.node_types.get(node as usize).map(String::as_str) == Some("foundation")
    }

    /// Find the index of the node nearest `point` whose type matches `pred`.
    fn nearest_node<P: Fn(&str) -> bool>(loaded: &LoadedScenario, point: Vec3, pred: P) -> u32 {
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for (i, n) in loaded.scenario.nodes.iter().enumerate() {
            let ty = loaded.node_types.get(i).map(String::as_str).unwrap_or("");
            if !pred(ty) {
                continue;
            }
            let dx = n.centroid.x - point.x;
            let dy = n.centroid.y - point.y;
            let dz = n.centroid.z - point.z;
            let d = dx * dx + dy * dy + dz * dz;
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    }

    fn bounds(loaded: &LoadedScenario) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut hi = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for n in &loaded.scenario.nodes {
            lo.x = lo.x.min(n.centroid.x);
            lo.y = lo.y.min(n.centroid.y);
            lo.z = lo.z.min(n.centroid.z);
            hi.x = hi.x.max(n.centroid.x);
            hi.y = hi.y.max(n.centroid.y);
            hi.z = hi.z.max(n.centroid.z);
        }
        (lo, hi)
    }

    // ── (a) Gravity stability ────────────────────────────────────────────────
    #[test]
    fn high_rise_stable_under_gravity() {
        let Some(loaded) = load() else { return };
        let m = run(&loaded, 240, None, 0);
        eprintln!("[gravity] {m:?}  bonds={}", loaded.scenario.bonds.len());
        assert_eq!(m.total_fractures, 0, "building must be stable under gravity");
        assert_eq!(m.final_actor_count, 1, "building must remain a single connected actor");
    }

    // Calibrated impact-force levels (Newtons) from the high_rise_sweep response
    // curve. These are stress-solver abstractions of the wrecking ball; the literal
    // ball mass/speed lives in the scene-pack JSON for the Rapier demos (where
    // contacts + splash damage govern). At these levels:
    //   LIGHT on infill   -> local panel blow-out, zero skeleton damage
    //   HEAVY on a column  -> local punch-through that severs some skeleton, but far
    //                         from a whole-structure cascade
    const LIGHT_FORCE: f32 = 1.0e7;
    const HEAVY_FORCE: f32 = 3.0e8;
    /// Even a big infill hit must stay well below a "whole face / whole building"
    /// shatter. A true glass cascade in this model is ~0.45 of all bonds (3e8 on
    /// infill); we require local damage to stay under this ceiling.
    const NO_GLASS_RATIO: f32 = 0.25;

    // ── (b) Light hit on infill -> local panel blow-out, skeleton intact ─────
    #[test]
    fn light_hit_blows_out_local_infill_only() {
        let Some(loaded) = load() else { return };
        let (lo, hi) = bounds(&loaded);
        // Aim at a mid-height infill panel on the -Z (front) face.
        let target_pt = Vec3::new(0.0, (lo.y + hi.y) * 0.5, lo.z);
        let node = nearest_node(&loaded, target_pt, |t| t == "infill");
        let pos = loaded.scenario.nodes[node as usize].centroid;
        let force = Vec3::new(0.0, 0.0, LIGHT_FORCE);
        let m = run(&loaded, 180, Some((node, pos, force)), 6);
        eprintln!("[light/infill] target_node={node} {m:?}");
        assert!(m.total_fractures > 0, "a light hit should detach some infill");
        assert_eq!(
            m.skeleton_fractures, 0,
            "a light infill hit must not break any skeleton (column/slab) bond"
        );
        assert_eq!(m.foundation_fractures, 0, "base must stay anchored");
        assert!(
            m.peak_actor_count <= 6,
            "infill should detach as a few local panels, not fragment the building (actors={})",
            m.peak_actor_count
        );
    }

    // ── (c) Heavy hit on a column -> bounded local punch-through, not glass ───
    #[test]
    fn heavy_hit_punches_bounded_hole() {
        let Some(loaded) = load() else { return };
        let (lo, _hi) = bounds(&loaded);
        // Aim at a low column (a couple storeys up) near the -Z face.
        let target_pt = Vec3::new(0.0, lo.y + 4.0, lo.z);
        let node = nearest_node(&loaded, target_pt, |t| t == "column");
        let pos = loaded.scenario.nodes[node as usize].centroid;
        let force = Vec3::new(0.0, 0.0, HEAVY_FORCE);
        let m = run(&loaded, 240, Some((node, pos, force)), 6);
        let total_bonds = loaded.scenario.bonds.len() as f32;
        let frac_ratio = m.total_fractures as f32 / total_bonds;
        eprintln!(
            "[heavy/column] target_node={node} {m:?} frac_ratio={:.3} total_bonds={}",
            frac_ratio, loaded.scenario.bonds.len()
        );
        assert!(m.total_fractures > 0, "a heavy hit should cause damage");
        assert!(
            m.skeleton_fractures > 0,
            "a heavy hit on a column should sever some skeleton"
        );
        // NOT a glass cascade: far below a whole-structure shatter.
        assert!(
            frac_ratio < NO_GLASS_RATIO,
            "heavy hit must punch a LOCAL hole, not shatter the whole structure (ratio {frac_ratio:.3})"
        );
    }

    // ── (c2) Even a big infill hit stays bounded (no whole-face glass) ───────
    #[test]
    fn big_infill_hit_does_not_glass_cascade() {
        let Some(loaded) = load() else { return };
        let (lo, hi) = bounds(&loaded);
        let node = nearest_node(&loaded, Vec3::new(0.0, (lo.y + hi.y) * 0.5, lo.z), |t| t == "infill");
        let pos = loaded.scenario.nodes[node as usize].centroid;
        // 1e8 N: an order of magnitude above a realistic ball; still must stay local.
        let force = Vec3::new(0.0, 0.0, 1.0e8);
        let m = run(&loaded, 200, Some((node, pos, force)), 6);
        let frac_ratio = m.total_fractures as f32 / loaded.scenario.bonds.len() as f32;
        eprintln!("[big/infill] {m:?} frac_ratio={frac_ratio:.3}");
        assert!(
            frac_ratio < NO_GLASS_RATIO,
            "a big infill hit must not blow out the whole structure (ratio {frac_ratio:.3})"
        );
    }

    // ── (d) Determinism ──────────────────────────────────────────────────────
    #[test]
    fn high_rise_is_deterministic() {
        let Some(loaded) = load() else { return };
        let (lo, hi) = bounds(&loaded);
        let node = nearest_node(&loaded, Vec3::new(0.0, (lo.y + hi.y) * 0.5, lo.z), |t| t == "infill");
        let pos = loaded.scenario.nodes[node as usize].centroid;
        let force = Vec3::new(0.0, 0.0, 2500.0 * 18.0 / 0.03);
        let m1 = run(&loaded, 120, Some((node, pos, force)), 2);
        let m2 = run(&loaded, 120, Some((node, pos, force)), 2);
        assert_eq!(m1.total_fractures, m2.total_fractures, "fracture count must be deterministic");
        assert_eq!(m1.final_actor_count, m2.final_actor_count, "actor count must be deterministic");
    }
}
