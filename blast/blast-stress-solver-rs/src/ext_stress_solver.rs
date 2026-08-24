//! High-level stress solver with NvBlast actor management and fracture support.

use crate::ffi;
use crate::types::*;

/// High-level stress solver that manages NvBlast families, actors, and fracture commands.
///
/// This is the primary API for most users. It wraps the C++ `ExtStressSolver` which internally
/// manages an NvBlast asset, family, and actors alongside the stress computation.
pub struct ExtStressSolver {
    handle: *mut ffi::ExtStressSolverHandle,
}

unsafe impl Send for ExtStressSolver {}
unsafe impl Sync for ExtStressSolver {}

impl ExtStressSolver {
    /// Create a solver from node and bond descriptors with the given settings.
    pub fn new(nodes: &[NodeDesc], bonds: &[BondDesc], settings: &SolverSettings) -> Option<Self> {
        if nodes.is_empty() || bonds.is_empty() {
            return None;
        }

        let ffi_nodes: Vec<ffi::FfiExtStressNodeDesc> = nodes
            .iter()
            .map(|n| ffi::FfiExtStressNodeDesc {
                centroid: n.centroid,
                mass: n.mass,
                volume: n.volume,
            })
            .collect();

        let ffi_bonds: Vec<ffi::FfiExtStressBondDesc> = bonds
            .iter()
            .map(|b| ffi::FfiExtStressBondDesc {
                centroid: b.centroid,
                normal: b.normal,
                area: b.area,
                node0: b.node0,
                node1: b.node1,
                // This crate exposes a single global strength; every bond uses
                // material 0 of the one-entry table built below.
                material: 0,
            })
            .collect();

        let ffi_settings = to_ffi_settings(settings);
        let ffi_materials = to_ffi_materials(settings);

        let handle = unsafe {
            ffi::ext_stress_solver_create(
                ffi_nodes.as_ptr(),
                ffi_nodes.len() as u32,
                ffi_bonds.as_ptr(),
                ffi_bonds.len() as u32,
                ffi_materials.as_ptr(),
                ffi_materials.len() as u32,
                &ffi_settings,
            )
        };

        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Per-bond decomposed stress (compression, tension, shear) in Pascals.
    ///
    /// Reads the solver's own view rather than recomputing it, so it can be
    /// compared directly against the material limits that decide fracture.
    /// Live per-bond health, indexed by asset bond index.
    ///
    /// Health crossing zero is the break. The fracture *command* stream is a
    /// damage stream: Blast issues a command every tick a bond is overstressed
    /// while its health is still positive, and the command's `health` field is
    /// the damage applied rather than what remains. Counting commands as breaks
    /// overcounts -- measured 1067 against a 546-bond tower.
    /// `out` is resized to `bond_count` and filled; the return is how many the
    /// solver actually wrote. Capacity is the caller's because `bond_count()`
    /// reports the post-graph-reduction count, which is not the asset bond
    /// indexing this array uses.
    pub fn bond_healths(&self, bond_count: usize, out: &mut Vec<f32>) -> usize {
        out.clear();
        out.resize(bond_count, 0.0);
        if bond_count == 0 {
            return 0;
        }
        let n = bond_count;
        let written = unsafe {
            ffi::ext_stress_solver_get_bond_healths(self.handle, out.as_mut_ptr(), n as u32)
        } as usize;
        out.truncate(written);
        written
    }

    pub fn bond_stresses(&self) -> Vec<BondStressResult> {
        let n = self.bond_count() as usize;
        let mut c = vec![0.0f32; n];
        let mut t = vec![0.0f32; n];
        let mut sh = vec![0.0f32; n];
        let written = unsafe {
            ffi::ext_stress_solver_get_bond_stresses(
                self.handle,
                c.as_mut_ptr(),
                t.as_mut_ptr(),
                sh.as_mut_ptr(),
                n as u32,
            )
        } as usize;
        (0..written)
            .map(|i| BondStressResult { compression: c[i], tension: t[i], shear: sh[i] })
            .collect()
    }

    /// Whether the solver is running island-aware (per-component) solves.
    pub fn island_aware(&self) -> bool {
        unsafe { ffi::ext_stress_solver_get_island_aware(self.handle) != 0 }
    }

    /// Enable island-aware solving: solve each disconnected component
    /// independently. Observationally identical to the whole-graph solve, and
    /// far cheaper once activity is localized.
    pub fn set_island_aware(&mut self, enabled: bool) {
        unsafe { ffi::ext_stress_solver_set_island_aware(self.handle, enabled as u8) }
    }

    /// Skip components whose velocity inputs are unchanged since their last
    /// solve and which already converged. A settled component re-solves the
    /// same frame its load changes, so it is paused, never frozen.
    pub fn set_skip_settled(&mut self, enabled: bool) {
        unsafe { ffi::ext_stress_solver_set_skip_settled(self.handle, enabled as u8) }
    }

    /// Components in the live graph, and how many were skipped last update.
    pub fn island_stats(&self) -> (u32, u32, u32) {
        unsafe {
            (
                ffi::ext_stress_solver_island_count(self.handle),
                ffi::ext_stress_solver_islands_total(self.handle),
                ffi::ext_stress_solver_islands_skipped(self.handle),
            )
        }
    }

    /// Update solver settings.
    pub fn set_settings(&mut self, settings: &SolverSettings) {
        let ffi_settings = to_ffi_settings(settings);
        unsafe { ffi::ext_stress_solver_set_settings(self.handle, &ffi_settings) }
    }

    /// Reset accumulated forces.
    pub fn reset(&mut self) {
        unsafe { ffi::ext_stress_solver_reset(self.handle) }
    }

    /// Apply a force to a specific node.
    pub fn add_force(&mut self, node_index: u32, position: Vec3, force: Vec3, mode: ForceMode) {
        unsafe {
            ffi::ext_stress_solver_add_force(
                self.handle,
                node_index,
                &position,
                &force,
                mode as u32,
            )
        }
    }

    /// Apply many forces in a single FFI crossing, mirroring [`add_force`] for
    /// each entry. `positions` and `forces` are flat `[x, y, z]` triples that
    /// run parallel to `node_indices` (entry `i` reads slice `3*i..3*i + 3`).
    /// `mode` is shared by every entry. Returns the number of entries applied.
    ///
    /// [`add_force`]: Self::add_force
    pub fn add_all_forces(
        &mut self,
        node_indices: &[u32],
        positions: &[f32],
        forces: &[f32],
        mode: ForceMode,
    ) -> u32 {
        let count = node_indices.len();
        if count == 0 {
            return 0;
        }
        debug_assert!(
            positions.len() >= count * 3 && forces.len() >= count * 3,
            "positions/forces must hold 3 floats per node index"
        );
        unsafe {
            ffi::ext_stress_solver_add_all_forces(
                self.handle,
                node_indices.as_ptr(),
                positions.as_ptr(),
                forces.as_ptr(),
                count as u32,
                mode as u32,
            )
        }
    }

    /// Apply gravity to all actors.
    pub fn add_gravity(&mut self, gravity: Vec3) {
        unsafe { ffi::ext_stress_solver_add_gravity(self.handle, &gravity) }
    }

    /// Apply gravity to a specific actor.
    pub fn add_actor_gravity(&mut self, actor_index: u32, gravity: Vec3) -> bool {
        unsafe { ffi::ext_stress_solver_add_actor_gravity(self.handle, actor_index, &gravity) != 0 }
    }

    /// Apply the centrifugal acceleration produced by an actor's angular movement.
    ///
    /// Mirrors NVIDIA Blast's default of applying scene gravity to static actors and
    /// centrifugal force to dynamic (tumbling) actors each frame, so spinning debris keeps
    /// accumulating stress and can secondary-fracture. `local_center_mass` and
    /// `local_angular_velocity` are expressed in the actor's local frame; the acceleration is
    /// applied to every node in the actor. Returns `false` if the actor no longer exists or no
    /// node received the acceleration.
    pub fn add_centrifugal_acceleration(
        &mut self,
        actor_index: u32,
        local_center_mass: Vec3,
        local_angular_velocity: Vec3,
    ) -> bool {
        unsafe {
            ffi::ext_stress_solver_add_centrifugal_acceleration(
                self.handle,
                actor_index,
                &local_center_mass,
                &local_angular_velocity,
            ) != 0
        }
    }

    /// Run one solver update (computes stresses from accumulated forces).
    pub fn update(&mut self) {
        unsafe { ffi::ext_stress_solver_update(self.handle) }
    }

    /// Number of bonds that exceeded their fatal stress limit after the last `update()`.
    pub fn overstressed_bond_count(&self) -> u32 {
        unsafe { ffi::ext_stress_solver_overstressed_bond_count(self.handle) }
    }

    /// Whether the solver converged in the last `update()`.
    pub fn converged(&self) -> bool {
        unsafe { ffi::ext_stress_solver_converged(self.handle) != 0 }
    }

    /// Linear error residual from the last solve.
    pub fn linear_error(&self) -> f32 {
        unsafe { ffi::ext_stress_solver_get_linear_error(self.handle) }
    }

    /// Angular error residual from the last solve.
    pub fn angular_error(&self) -> f32 {
        unsafe { ffi::ext_stress_solver_get_angular_error(self.handle) }
    }

    /// Number of actors currently in the family.
    pub fn actor_count(&self) -> u32 {
        unsafe { ffi::ext_stress_solver_actor_count(self.handle) }
    }

    /// Total number of graph nodes.
    pub fn node_count(&self) -> u32 {
        unsafe { ffi::ext_stress_solver_graph_node_count(self.handle) }
    }

    /// Total number of bonds.
    pub fn bond_count(&self) -> u32 {
        unsafe { ffi::ext_stress_solver_bond_count(self.handle) }
    }

    /// Collect the current actor table.
    pub fn actors(&self) -> Vec<Actor> {
        let actor_count = self.actor_count();
        if actor_count == 0 {
            return Vec::new();
        }
        let node_count = self.node_count();

        let mut actor_buffer = vec![
            ffi::FfiExtStressActor {
                actor_index: u32::MAX,
                nodes: std::ptr::null(),
                node_count: 0,
            };
            actor_count as usize
        ];
        let mut nodes_buffer = vec![0u32; node_count as usize];
        let mut out_actor_count = 0u32;
        let mut out_node_count = 0u32;

        unsafe {
            ffi::ext_stress_solver_collect_actors(
                self.handle,
                actor_buffer.as_mut_ptr(),
                actor_count,
                nodes_buffer.as_mut_ptr(),
                node_count,
                &mut out_actor_count,
                &mut out_node_count,
            );
        }

        let mut result = Vec::with_capacity(out_actor_count as usize);
        for i in 0..out_actor_count as usize {
            let ffi_actor = &actor_buffer[i];
            let nodes = if !ffi_actor.nodes.is_null() && ffi_actor.node_count > 0 {
                let offset = unsafe { ffi_actor.nodes.offset_from(nodes_buffer.as_ptr()) } as usize;
                nodes_buffer[offset..offset + ffi_actor.node_count as usize].to_vec()
            } else {
                Vec::new()
            };
            result.push(Actor {
                actor_index: ffi_actor.actor_index,
                nodes,
            });
        }
        result
    }

    /// Collect just `(actor_index, first_node_index)` for each actor — a lighter variant
    /// of [`actors`](Self::actors) for per-frame hot paths (oriented-gravity / excess-force
    /// application) that need only one representative node per actor. Avoids the per-actor
    /// `Vec<u32>` allocations `actors()` makes (one allocation per actor, every frame).
    pub fn collect_actor_reps(&self) -> Vec<(u32, u32)> {
        let actor_count = self.actor_count();
        if actor_count == 0 {
            return Vec::new();
        }
        let node_count = self.node_count();

        let mut actor_buffer = vec![
            ffi::FfiExtStressActor {
                actor_index: u32::MAX,
                nodes: std::ptr::null(),
                node_count: 0,
            };
            actor_count as usize
        ];
        let mut nodes_buffer = vec![0u32; node_count as usize];
        let mut out_actor_count = 0u32;
        let mut out_node_count = 0u32;

        unsafe {
            ffi::ext_stress_solver_collect_actors(
                self.handle,
                actor_buffer.as_mut_ptr(),
                actor_count,
                nodes_buffer.as_mut_ptr(),
                node_count,
                &mut out_actor_count,
                &mut out_node_count,
            );
        }

        let mut reps = Vec::with_capacity(out_actor_count as usize);
        for i in 0..out_actor_count as usize {
            let a = &actor_buffer[i];
            if !a.nodes.is_null() && a.node_count > 0 {
                let offset = unsafe { a.nodes.offset_from(nodes_buffer.as_ptr()) } as usize;
                reps.push((a.actor_index, nodes_buffer[offset]));
            }
        }
        reps
    }

    /// Generate fracture commands for all actors with overstressed bonds.
    pub fn generate_fracture_commands(&self) -> Vec<FractureCommand> {
        let actor_count = self.actor_count();
        let bond_count = self.bond_count();
        if actor_count == 0 || bond_count == 0 {
            return Vec::new();
        }

        let mut command_buffer = vec![
            ffi::FfiExtStressFractureCommands {
                actor_index: u32::MAX,
                bond_fractures: std::ptr::null_mut(),
                bond_fracture_count: 0,
            };
            actor_count as usize
        ];
        let mut bond_buffer = vec![ffi::FfiExtStressBondFracture::default(); bond_count as usize];
        let mut out_command_count = 0u32;
        let mut out_bond_count = 0u32;

        unsafe {
            ffi::ext_stress_solver_generate_fracture_commands_per_actor(
                self.handle,
                command_buffer.as_mut_ptr(),
                actor_count,
                bond_buffer.as_mut_ptr(),
                bond_count,
                &mut out_command_count,
                &mut out_bond_count,
            );
        }

        let mut result = Vec::with_capacity(out_command_count as usize);
        for i in 0..out_command_count as usize {
            let cmd = &command_buffer[i];
            let fractures = if !cmd.bond_fractures.is_null() && cmd.bond_fracture_count > 0 {
                let offset =
                    unsafe { cmd.bond_fractures.offset_from(bond_buffer.as_ptr()) } as usize;
                bond_buffer[offset..offset + cmd.bond_fracture_count as usize]
                    .iter()
                    .map(|f| BondFracture {
                        userdata: f.userdata,
                        node_index0: f.node_index0,
                        node_index1: f.node_index1,
                        health: f.health,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            result.push(FractureCommand {
                actor_index: cmd.actor_index,
                bond_fractures: fractures,
            });
        }
        result
    }

    /// Apply fracture commands and return split events.
    pub fn apply_fracture_commands(&mut self, commands: &[FractureCommand]) -> Vec<SplitEvent> {
        if commands.is_empty() {
            return Vec::new();
        }

        // Flatten bond fractures into a contiguous buffer
        let total_bonds: usize = commands.iter().map(|c| c.bond_fractures.len()).sum();
        let mut flat_bonds = vec![ffi::FfiExtStressBondFracture::default(); total_bonds];
        let mut ffi_commands = Vec::with_capacity(commands.len());
        let mut offset = 0usize;

        for cmd in commands {
            for (i, f) in cmd.bond_fractures.iter().enumerate() {
                flat_bonds[offset + i] = ffi::FfiExtStressBondFracture {
                    userdata: f.userdata,
                    node_index0: f.node_index0,
                    node_index1: f.node_index1,
                    health: f.health,
                };
            }
            ffi_commands.push(ffi::FfiExtStressFractureCommands {
                actor_index: cmd.actor_index,
                bond_fractures: if cmd.bond_fractures.is_empty() {
                    std::ptr::null_mut()
                } else {
                    unsafe { flat_bonds.as_mut_ptr().add(offset) }
                },
                bond_fracture_count: cmd.bond_fractures.len() as u32,
            });
            offset += cmd.bond_fractures.len();
        }

        // Allocate output buffers
        let node_count = self.node_count() as usize;
        let max_events = commands.len();
        let max_children = node_count;

        let mut events_buffer = vec![
            ffi::FfiExtStressSplitEvent {
                parent_actor_index: u32::MAX,
                children: std::ptr::null_mut(),
                child_count: 0,
            };
            max_events
        ];
        let mut child_buffer = vec![
            ffi::FfiExtStressActor {
                actor_index: u32::MAX,
                nodes: std::ptr::null(),
                node_count: 0,
            };
            max_children
        ];
        let mut nodes_buffer = vec![0u32; node_count];
        let mut out_event_count = 0u32;
        let mut out_child_count = 0u32;
        let mut out_node_count = 0u32;

        unsafe {
            ffi::ext_stress_solver_apply_fracture_commands(
                self.handle,
                ffi_commands.as_ptr(),
                ffi_commands.len() as u32,
                events_buffer.as_mut_ptr(),
                max_events as u32,
                child_buffer.as_mut_ptr(),
                max_children as u32,
                &mut out_event_count,
                &mut out_child_count,
                nodes_buffer.as_mut_ptr(),
                node_count as u32,
                &mut out_node_count,
            );
        }

        // Parse results
        let mut result = Vec::with_capacity(out_event_count as usize);
        for i in 0..out_event_count as usize {
            let evt = &events_buffer[i];
            let mut children = Vec::with_capacity(evt.child_count as usize);

            if !evt.children.is_null() && evt.child_count > 0 {
                let child_offset =
                    unsafe { evt.children.offset_from(child_buffer.as_ptr()) } as usize;
                for ci in 0..evt.child_count as usize {
                    let child = &child_buffer[child_offset + ci];
                    let nodes = if !child.nodes.is_null() && child.node_count > 0 {
                        let node_off =
                            unsafe { child.nodes.offset_from(nodes_buffer.as_ptr()) } as usize;
                        nodes_buffer[node_off..node_off + child.node_count as usize].to_vec()
                    } else {
                        Vec::new()
                    };
                    children.push(SplitChild {
                        actor_index: child.actor_index,
                        nodes,
                    });
                }
            }

            result.push(SplitEvent {
                parent_actor_index: evt.parent_actor_index,
                children,
            });
        }
        result
    }

    /// Get excess forces for an actor that separated from the structure.
    /// Returns `(force, torque)` if the actor exists.
    pub fn get_excess_forces(&self, actor_index: u32, com: Vec3) -> Option<(Vec3, Vec3)> {
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        let ok = unsafe {
            ffi::ext_stress_solver_get_excess_forces(
                self.handle,
                actor_index,
                &com,
                &mut force,
                &mut torque,
            )
        };
        if ok != 0 {
            Some((force, torque))
        } else {
            None
        }
    }
}

impl Drop for ExtStressSolver {
    fn drop(&mut self) {
        unsafe { ffi::ext_stress_solver_destroy(self.handle) }
    }
}

fn to_ffi_settings(s: &SolverSettings) -> ffi::FfiExtStressSolverSettingsDesc {
    ffi::FfiExtStressSolverSettingsDesc {
        max_solver_iterations_per_frame: s.max_solver_iterations_per_frame,
        graph_reduction_level: s.graph_reduction_level,
    }
}

/// Stress limits moved from settings to a per-bond material table. This crate
/// still exposes one global strength, so it builds a single-entry table and
/// leaves every bond on material 0 — identical behavior to before the split.
fn to_ffi_materials(s: &SolverSettings) -> [ffi::FfiExtStressMaterialDesc; 1] {
    [ffi::FfiExtStressMaterialDesc {
        compression_elastic_limit: s.compression_elastic_limit,
        compression_fatal_limit: s.compression_fatal_limit,
        tension_elastic_limit: s.tension_elastic_limit,
        tension_fatal_limit: s.tension_fatal_limit,
        shear_elastic_limit: s.shear_elastic_limit,
        shear_fatal_limit: s.shear_fatal_limit,
        // Crush stays off here: `crush_cap_pressure == 0.0` disables it. These
        // fields exist so the struct matches the C ABI byte-for-byte; opting
        // into crushing goes through `set_materials`, not `SolverSettings`.
        ..ffi::FfiExtStressMaterialDesc::default()
    }]
}
