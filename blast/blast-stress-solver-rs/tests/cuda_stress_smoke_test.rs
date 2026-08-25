//! Opt-in regression test for the CUDA stress solver's settled-island skipping.
//!
//! **Why it is opt-in.** The CUDA solver is not part of this crate's build: it
//! is one `.cu` compiled by the consumer (see `physx-bridge/build.rs` in
//! vibe-land), and it needs `nvcc` plus a real GPU. Wiring it into the default
//! suite would make the suite unbuildable on a machine without CUDA. Making it
//! auto-skip instead would be worse -- a GPU test that quietly downgrades to a
//! pass is how a GPU path stays broken for weeks.
//!
//! **What it proves.** That `ExtStressGpuSolveParams::skipSettledIslands` is
//! observationally free: three solvers are driven over one scripted velocity
//! stream, two with skipping off and one with it on, and the skipping run must
//! not differ from a skip-off run by more than two skip-off runs differ from
//! each other. The GPU accumulates node loads with float `atomicAdd`, whose
//! ordering is not reproducible, so that noise floor -- not a constant
//! tolerance -- is the only honest yardstick. Broken-bond counts must match
//! exactly, every tick.
//!
//! ```text
//! cargo test -p blast-stress-solver --features cuda-stress-smoke \
//!     --test cuda_stress_smoke_test -- --nocapture
//! ```

#![cfg(feature = "cuda-stress-smoke")]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn settled_island_skipping_does_not_change_the_physics() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../source/sdk/extensions/stressgpu/test/build_and_run.sh");
    assert!(
        script.is_file(),
        "missing the CUDA equivalence harness at {}",
        script.display()
    );
    let output = Command::new("bash")
        .arg(&script)
        .output()
        .expect("run the CUDA settled-skip equivalence harness");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stdout}{stderr}");
    assert!(
        output.status.success(),
        "settled-island skipping changed the physics; harness output above"
    );
}
