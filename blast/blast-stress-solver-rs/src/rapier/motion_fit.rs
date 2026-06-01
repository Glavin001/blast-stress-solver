//! Pure math kernels for fitting rigid-body motion to a set of weighted node
//! samples during a fracture split.
//!
//! These were lifted out of [`super::body_tracker`] so they can be unit- and
//! property-tested in isolation (no Rapier world, no FFI). The arithmetic is a
//! byte-for-byte move of the original inline code — see
//! `body_tracker::fit_child_motion` / `child_world_center_of_mass`.
//!
//! Note on the known split bug: these kernels are *correct* when the chosen
//! `center` equals the mass-weighted centroid of the sample points (which is
//! exactly what the caller passes). The "sudden movement after destruction"
//! bug does NOT live here — it is the divergence between this centroid model
//! and Rapier's collider-derived centre of mass, which only manifests at
//! integration level (see `tests/kinematic_invariants_test.rs`). The properties
//! below therefore act as regression guards and ill-conditioning documentation.

use crate::types::Vec3;

/// Solve the symmetric 3x3 system `M · x = rhs` by explicit inversion.
///
/// Returns `None` when `M` is (numerically) singular — `|det| <= 1e-6` — so the
/// caller can fall back to a known-good angular velocity instead of inverting an
/// ill-conditioned matrix (which would yield a spurious, often huge, result).
pub(crate) fn solve_symmetric_3x3(m: [[f32; 3]; 3], rhs: [f32; 3]) -> Option<Vec3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() <= 1.0e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ];
    Some(Vec3::new(
        inv[0][0] * rhs[0] + inv[0][1] * rhs[1] + inv[0][2] * rhs[2],
        inv[1][0] * rhs[0] + inv[1][1] * rhs[1] + inv[1][2] * rhs[2],
        inv[2][0] * rhs[0] + inv[2][1] * rhs[1] + inv[2][2] * rhs[2],
    ))
}

/// Mass-weighted centroid of `(position, mass)` samples.
///
/// Mirrors `body_tracker::child_world_center_of_mass`: only strictly-positive
/// masses contribute, and `None` is returned when no mass is present so the
/// caller can fall back to the unweighted centroid.
pub(crate) fn weighted_center_of_mass(samples: &[(Vec3, f32)]) -> Option<Vec3> {
    let mut weighted = Vec3::ZERO;
    let mut total = 0.0f32;
    for &(position, mass) in samples {
        if mass > 0.0 {
            weighted += position * mass;
            total += mass;
        }
    }
    if total <= f32::EPSILON {
        None
    } else {
        Some(weighted / total)
    }
}

/// Result of fitting a single rigid motion to a cloud of node samples.
pub(crate) struct RigidMotionFit {
    /// Mass-weighted mean velocity (the body's centre-of-mass velocity).
    pub linvel: Vec3,
    /// Best-fit angular velocity about `center`, or `None` when the linear
    /// system is singular (degenerate point cloud).
    pub angvel: Option<Vec3>,
}

/// Least-squares fit of a rigid-body velocity field to `(point, velocity, mass)`
/// samples, expressed about `center`.
///
/// `linvel` is the mass-weighted mean velocity; `angvel` solves
/// `I · ω = Σ mᵢ (rᵢ × (vᵢ − linvel))` with `rᵢ = pointᵢ − center` and `I` the
/// sample inertia tensor about `center`. This is exact (recovers the true
/// `(v, ω)`) iff `center` is the mass-weighted centroid of the points — which is
/// what `body_tracker` passes.
pub(crate) fn fit_rigid_motion(samples: &[(Vec3, Vec3, f32)], center: Vec3) -> Option<RigidMotionFit> {
    let mut linvel_sum = Vec3::ZERO;
    let mut total_mass = 0.0f32;
    for &(_, velocity, mass) in samples {
        linvel_sum += velocity * mass;
        total_mass += mass;
    }
    if samples.is_empty() || total_mass <= f32::EPSILON {
        return None;
    }
    let linvel = linvel_sum / total_mass;

    let mut normal = [[0.0f32; 3]; 3];
    let mut rhs = [0.0f32; 3];
    for &(point, velocity, mass) in samples {
        let r = point - center;
        let v_rel = velocity - linvel;
        let r2 = r.magnitude_squared();
        let rr = [
            [r.x * r.x, r.x * r.y, r.x * r.z],
            [r.y * r.x, r.y * r.y, r.y * r.z],
            [r.z * r.x, r.z * r.y, r.z * r.z],
        ];
        for row in 0..3 {
            for col in 0..3 {
                normal[row][col] += mass * ((if row == col { r2 } else { 0.0 }) - rr[row][col]);
            }
        }
        let cross = r.cross(v_rel);
        rhs[0] += mass * cross.x;
        rhs[1] += mass * cross.y;
        rhs[2] += mass * cross.z;
    }

    Some(RigidMotionFit {
        linvel,
        angvel: solve_symmetric_3x3(normal, rhs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn vdist(a: Vec3, b: Vec3) -> f32 {
        (a - b).magnitude()
    }

    /// A fixed, well-conditioned (non-degenerate, non-collinear) point cloud so
    /// the recovery properties never hit a near-singular inertia matrix. This is
    /// deliberate: the *math* is exercised by randomising the motion, not the
    /// geometry, which keeps the property tests deterministic and non-flaky.
    fn tetra(scale: f32) -> [Vec3; 4] {
        [
            Vec3::new(1.0, 1.0, 1.0) * scale,
            Vec3::new(1.0, -1.0, -1.0) * scale,
            Vec3::new(-1.0, 1.0, -1.0) * scale,
            Vec3::new(-1.0, -1.0, 1.0) * scale,
        ]
    }

    proptest! {
        // ---- solve_symmetric_3x3: round-trips a well-conditioned SPD system ----
        #[test]
        fn solve_round_trips_spd(
            a in prop::array::uniform9(-2.0f32..2.0),
            x in prop::array::uniform3(-5.0f32..5.0),
        ) {
            // M = AᵀA + ε I is symmetric positive-definite (well conditioned for ε large).
            let am = [[a[0], a[1], a[2]], [a[3], a[4], a[5]], [a[6], a[7], a[8]]];
            let mut m = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = 0.0;
                    for k in 0..3 {
                        s += am[k][i] * am[k][j];
                    }
                    m[i][j] = s + if i == j { 1.0 } else { 0.0 };
                }
            }
            let rhs = [
                m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2],
                m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2],
                m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2],
            ];
            let solved = solve_symmetric_3x3(m, rhs).expect("SPD system must be solvable");
            let expected = Vec3::new(x[0], x[1], x[2]);
            prop_assert!(
                vdist(solved, expected) <= 1.0e-2 * (1.0 + expected.magnitude()),
                "solve mismatch: got {:?} expected {:?}",
                solved,
                expected
            );
        }

        // ---- fit_rigid_motion: exact recovery when center == mass-weighted centroid ----
        #[test]
        fn fit_recovers_rigid_motion(
            v in prop::array::uniform3(-8.0f32..8.0),
            w in prop::array::uniform3(-8.0f32..8.0),
            com in prop::array::uniform3(-6.0f32..6.0),
            scale in 0.4f32..3.0,
        ) {
            let linvel_true = Vec3::new(v[0], v[1], v[2]);
            let omega_true = Vec3::new(w[0], w[1], w[2]);
            let true_com = Vec3::new(com[0], com[1], com[2]);

            // Equal masses => mass-weighted centroid is the geometric centroid (= true_com,
            // since the tetra is centred at the origin and we shift it by true_com).
            let points: Vec<Vec3> = tetra(scale).iter().map(|&p| p + true_com).collect();
            let samples: Vec<(Vec3, Vec3, f32)> = points
                .iter()
                .map(|&p| (p, linvel_true + omega_true.cross(p - true_com), 1.0))
                .collect();

            let fit = fit_rigid_motion(&samples, true_com).expect("non-empty samples");
            let angvel = fit.angvel.expect("non-degenerate cloud must yield an angvel");

            prop_assert!(
                vdist(fit.linvel, linvel_true) <= 1.0e-2 * (1.0 + linvel_true.magnitude()),
                "linvel mismatch: got {:?} expected {:?}",
                fit.linvel,
                linvel_true
            );
            prop_assert!(
                vdist(angvel, omega_true) <= 1.0e-2 * (1.0 + omega_true.magnitude()),
                "angvel mismatch: got {:?} expected {:?}",
                angvel,
                omega_true
            );
        }

        // ---- fit_rigid_motion: linvel is the mass-weighted mean regardless of geometry ----
        #[test]
        fn fit_linvel_is_mass_weighted_mean(
            vs in prop::collection::vec(prop::array::uniform3(-5.0f32..5.0), 1..6),
            ms in prop::collection::vec(0.2f32..5.0, 1..6),
        ) {
            let n = vs.len().min(ms.len());
            let samples: Vec<(Vec3, Vec3, f32)> = (0..n)
                .map(|i| {
                    (
                        Vec3::new(i as f32, 0.0, 0.0),
                        Vec3::new(vs[i][0], vs[i][1], vs[i][2]),
                        ms[i],
                    )
                })
                .collect();
            let mut wsum = Vec3::ZERO;
            let mut msum = 0.0f32;
            for &(_, vel, m) in &samples {
                wsum += vel * m;
                msum += m;
            }
            let expected = wsum / msum;
            let fit = fit_rigid_motion(&samples, Vec3::ZERO).expect("non-empty");
            prop_assert!(vdist(fit.linvel, expected) <= 1.0e-3 * (1.0 + expected.magnitude()));
        }
    }

    // ---- Degenerate-geometry guards: must not panic or NaN, must report singular ----

    #[test]
    fn single_point_cloud_is_singular() {
        let samples = [(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, 0.0), 1.0)];
        let center = Vec3::new(1.0, 2.0, 3.0);
        let fit = fit_rigid_motion(&samples, center).expect("non-empty");
        // r = 0 for the only sample => zero inertia tensor => singular => no angvel.
        assert!(fit.angvel.is_none(), "single-point cloud must be singular");
        assert!(fit.linvel.x.is_finite() && fit.linvel.y.is_finite() && fit.linvel.z.is_finite());
    }

    #[test]
    fn collinear_cloud_does_not_produce_nan() {
        // Three points on the x axis: the inertia tensor is rank-deficient about x.
        let samples = [
            (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0),
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 1.0),
        ];
        let fit = fit_rigid_motion(&samples, Vec3::ZERO).expect("non-empty");
        if let Some(w) = fit.angvel {
            assert!(
                w.x.is_finite() && w.y.is_finite() && w.z.is_finite(),
                "collinear fit must not produce NaN/Inf: {w:?}"
            );
        }
    }

    #[test]
    fn empty_or_massless_center_of_mass_is_none() {
        assert!(weighted_center_of_mass(&[]).is_none());
        assert!(weighted_center_of_mass(&[(Vec3::new(1.0, 0.0, 0.0), 0.0)]).is_none());
        let com = weighted_center_of_mass(&[
            (Vec3::new(0.0, 0.0, 0.0), 1.0),
            (Vec3::new(2.0, 0.0, 0.0), 3.0),
        ])
        .unwrap();
        // weighted toward the heavier node: (0*1 + 2*3)/4 = 1.5
        assert!((com.x - 1.5).abs() < 1e-6, "got {com:?}");
    }
}
