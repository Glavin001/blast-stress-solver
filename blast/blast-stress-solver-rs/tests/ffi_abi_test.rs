//! Pins the Rust `#[repr(C)]` mirrors against the C ABI's own `sizeof`.
//!
//! These descriptors cross the FFI boundary by pointer, so a field-count drift
//! is not a compile error — it is an out-of-bounds read at runtime. That is not
//! hypothetical: `FfiExtStressMaterialDesc` shipped carrying only the six
//! stress limits while the C `ExtStressMaterialDesc` also carried nine crush
//! fields, so every `ext_stress_solver_create` had C read 60 bytes out of a
//! 24-byte Rust allocation and take the crush parameters — including
//! `crush_cap_pressure`, which is what *enables* crushing — from whatever
//! happened to follow in memory.
//!
//! The C bridge already exports `ext_stress_sizeof_*` for exactly this purpose
//! (the JS binding uses them to size heap buffers). This asserts the Rust side
//! against the same source of truth.

use blast_stress_solver::abi::{c_abi_sizes, c_abi_version, rust_mirror_sizes, EXPECTED_ABI_VERSION};

#[test]
fn repr_c_mirrors_match_the_c_abi() {
    let c = c_abi_sizes();
    let r = rust_mirror_sizes();

    assert_eq!(
        r.material_desc, c.material_desc,
        "ExtStressMaterialDesc drifted: Rust {} bytes vs C {} bytes. This is \
         passed by pointer to ext_stress_solver_create, so a mismatch is an \
         out-of-bounds read, not a compile error.",
        r.material_desc, c.material_desc
    );
    assert_eq!(
        r.settings_desc, c.settings_desc,
        "ExtStressSolverSettingsDesc drifted: Rust {} vs C {}",
        r.settings_desc, c.settings_desc
    );
    assert_eq!(
        r.bond_fracture, c.bond_fracture,
        "ExtStressBondFracture drifted: Rust {} vs C {}",
        r.bond_fracture, c.bond_fracture
    );
    assert_eq!(
        r.node_desc, c.node_desc,
        "ExtStressNodeDesc drifted: Rust {} vs C {}",
        r.node_desc, c.node_desc
    );
    assert_eq!(
        r.bond_desc, c.bond_desc,
        "ExtStressBondDesc drifted: Rust {} vs C {}",
        r.bond_desc, c.bond_desc
    );
}

#[test]
fn material_desc_still_carries_the_crush_fields() {
    // Guards the specific regression: the six-limit-only struct was 24 bytes.
    // If this ever reads 24 again, the crush fields have been dropped and
    // `set_materials` would silently stop configuring comminution.
    let c = c_abi_sizes().material_desc;
    assert!(
        c > 24,
        "C material desc is {c} bytes — the crush fields appear to be gone"
    );
}

#[test]
fn the_old_six_field_layout_would_be_rejected() {
    // Demonstrates this suite's discriminating power: reconstruct the exact
    // struct that shipped before the fix and show it does NOT match the C ABI.
    // Without this, `repr_c_mirrors_match_the_c_abi` passing would only prove
    // the current struct is self-consistent, not that it catches the drift.
    #[repr(C)]
    struct PreFixMaterialDesc {
        compression_elastic_limit: f32,
        compression_fatal_limit: f32,
        tension_elastic_limit: f32,
        tension_fatal_limit: f32,
        shear_elastic_limit: f32,
        shear_fatal_limit: f32,
    }

    let c = c_abi_sizes().material_desc;
    let stale = std::mem::size_of::<PreFixMaterialDesc>();
    assert_eq!(stale, 24, "the pre-fix struct was six f32s");
    assert_ne!(
        stale, c,
        "the pre-fix 24-byte layout must not match the {c}-byte C struct — \
         if these are equal the ABI probe is not discriminating"
    );
    eprintln!("[abi] C material_desc = {c} bytes; pre-fix Rust mirror = {stale} bytes \
(a {} byte out-of-bounds read per material)", c - stale);
}

#[test]
fn abi_version_matches_the_linked_bridge() {
    // Catches the case where the crate is compiled against native sources from
    // a different checkout -- the exact hazard for consumers that point a
    // BLAST_ROOT at an out-of-tree copy of `blast/`.
    assert_eq!(
        c_abi_version(),
        EXPECTED_ABI_VERSION,
        "C bridge reports ABI v{} but this crate expects v{}. The Rust crate \
         and the native sources are from different checkouts.",
        c_abi_version(),
        EXPECTED_ABI_VERSION
    );
}
