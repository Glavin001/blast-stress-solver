use rapier3d::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ResimulationOptions {
    pub enabled: bool,
    pub max_passes: usize,
}

impl Default for ResimulationOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_passes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BodySnapshot {
    handle: RigidBodyHandle,
    position: Isometry<Real>,
    linvel: Vector<Real>,
    angvel: AngVector<Real>,
    linear_damping: Real,
    angular_damping: Real,
    sleeping: bool,
    enabled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct BodySnapshots {
    bodies: Vec<BodySnapshot>,
}

impl BodySnapshots {
    pub fn capture(set: &RigidBodySet) -> Self {
        let mut snapshots = Self::default();
        snapshots.capture_into(set);
        snapshots
    }

    /// Refill this snapshot from `set`, reusing the existing buffer. When the (non-fixed)
    /// body count is stable this performs **no heap allocation** — the resim-friendly way
    /// to snapshot every frame at scale (thousands of debris bodies), instead of allocating
    /// a fresh `Vec` via [`capture`](Self::capture) each rollback.
    pub fn capture_into(&mut self, set: &RigidBodySet) {
        self.bodies.clear();
        self.bodies.extend(set.iter().filter_map(|(handle, body)| {
            if body.is_fixed() {
                None
            } else {
                Some(BodySnapshot {
                    handle,
                    position: *body.position(),
                    linvel: *body.linvel(),
                    angvel: *body.angvel(),
                    linear_damping: body.linear_damping(),
                    angular_damping: body.angular_damping(),
                    sleeping: body.is_sleeping(),
                    enabled: body.is_enabled(),
                })
            }
        }));
    }

    pub fn restore(&self, set: &mut RigidBodySet) {
        for snapshot in &self.bodies {
            let Some(body) = set.get_mut(snapshot.handle) else {
                continue;
            };
            body.set_enabled(snapshot.enabled);
            body.set_position(snapshot.position, true);
            body.set_linvel(snapshot.linvel, true);
            body.set_angvel(snapshot.angvel, true);
            body.set_linear_damping(snapshot.linear_damping);
            body.set_angular_damping(snapshot.angular_damping);
            body.reset_forces(true);
            body.reset_torques(true);
            if snapshot.sleeping {
                body.sleep();
            } else {
                body.wake_up(true);
            }
        }
    }
}
