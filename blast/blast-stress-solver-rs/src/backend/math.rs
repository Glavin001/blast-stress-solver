//! Engine-neutral pose math.
//!
//! The core owns these so the pipeline never names an engine's vector type.
//! Adapters convert at the boundary (one memcpy-shaped conversion per batch,
//! not per call).

use crate::types::Vec3;

/// Unit quaternion, `w` last to match every engine this targets.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Rotate a vector by this quaternion.
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
    }

    /// Rotate a vector by the inverse of this quaternion.
    pub fn rotate_inverse(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    pub fn conjugate(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }
}

/// Rigid transform: rotation then translation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Pose {
    pub const IDENTITY: Self = Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY };

    pub const fn from_translation(translation: Vec3) -> Self {
        Self { translation, rotation: Quat::IDENTITY }
    }

    pub fn new(translation: Vec3, rotation: Quat) -> Self {
        Self { translation, rotation }
    }

    /// Map a point from this frame into world space.
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.rotation.rotate(p) + self.translation
    }

    /// Map a world point into this frame.
    pub fn inverse_transform_point(self, p: Vec3) -> Vec3 {
        self.rotation.rotate_inverse(p - self.translation)
    }
}
