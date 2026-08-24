use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let blast_root = manifest_dir.parent().expect("blast dir");
    let repo_root = blast_root.parent().expect("repo root");

    let target = env::var("TARGET").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let is_wasm = target_arch == "wasm32";
    let is_wasm_unknown = is_wasm && target == "wasm32-unknown-unknown";

    let blast = repo_root.join("blast");

    // The PhysX backend is an independent native unit: it links the PhysX SDK
    // and shares no translation units with the Blast bridge below.
    #[cfg(feature = "physx")]
    build_physx_backend(&blast);
    let ffi_dir = blast.join("rust_stress_example/ffi");

    // --- Source files ---
    let bridge_sources = [
        ffi_dir.join("stress_bridge.cpp"),
        ffi_dir.join("ext_stress_bridge.cpp"),
    ];

    let blast_sources = [
        // Core stress solver
        blast.join("source/shared/stress_solver/stress.cpp"),
        // SDK common
        blast.join("source/sdk/common/NvBlastAssert.cpp"),
        blast.join("source/sdk/common/NvBlastAtomic.cpp"),
        blast.join("source/sdk/common/NvBlastTime.cpp"),
        blast.join("source/sdk/common/NvBlastTimers.cpp"),
        // SDK globals
        blast.join("source/sdk/globals/NvBlastGlobals.cpp"),
        blast.join("source/sdk/globals/NvBlastInternalProfiler.cpp"),
        // SDK lowlevel
        blast.join("source/sdk/lowlevel/NvBlastAsset.cpp"),
        blast.join("source/sdk/lowlevel/NvBlastAssetHelper.cpp"),
        blast.join("source/sdk/lowlevel/NvBlastFamily.cpp"),
        blast.join("source/sdk/lowlevel/NvBlastFamilyGraph.cpp"),
        blast.join("source/sdk/lowlevel/NvBlastActor.cpp"),
        blast.join("source/sdk/lowlevel/NvBlastActorSerializationBlock.cpp"),
        // Extension: stress solver
        blast.join("source/sdk/extensions/stress/NvBlastExtStressSolver.cpp"),
    ];

    // --- Rerun-if-changed ---
    for src in bridge_sources.iter().chain(blast_sources.iter()) {
        println!("cargo:rerun-if-changed={}", src.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        ffi_dir.join("stress_bridge.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        ffi_dir.join("ext_stress_bridge.h").display()
    );

    // --- Build ---
    let mut build = cc::Build::new();
    build.cpp(true);

    // --- WebAssembly (wasm32-unknown-unknown) support ---
    //
    // The upstream crate targets native + (aspirationally) wasm32. When
    // cargo is invoked with `--target wasm32-unknown-unknown` we need to
    // (a) compile the C++ against the wasi libc/libc++ that ships with
    // clang-18+ (Ubuntu: `libc++-18-dev-wasm32` + `wasi-libc`), and
    // (b) override the clang target so it picks up those headers —
    // clang refuses to use the wasi sysroot when the triple is
    // `wasm32-unknown-unknown`.
    //
    // Users can force specific paths via the `BLAST_WASM_SYSROOT` and
    // `BLAST_WASM_CXX_INCLUDE` env vars; otherwise we probe a few
    // common locations. This logic is no-op on native targets.
    if is_wasm_unknown {
        build.target("wasm32-wasi");

        let sysroot = env::var("BLAST_WASM_SYSROOT")
            .ok()
            .or_else(|| probe_wasi_sysroot().map(|p| p.display().to_string()));
        let cxx_include = env::var("BLAST_WASM_CXX_INCLUDE")
            .ok()
            .or_else(|| probe_libcxx_wasm_include().map(|p| p.display().to_string()));

        if let Some(sysroot) = &sysroot {
            println!("cargo:rerun-if-env-changed=BLAST_WASM_SYSROOT");
            build.flag(&format!("--sysroot={sysroot}"));
        }
        if let Some(cxx_include) = &cxx_include {
            println!("cargo:rerun-if-env-changed=BLAST_WASM_CXX_INCLUDE");
            build.flag("-isystem");
            build.flag(cxx_include);
        }

        if sysroot.is_none() || cxx_include.is_none() {
            println!(
                "cargo:warning=blast-stress-solver: wasi sysroot or libc++ headers not found — \
                 set BLAST_WASM_SYSROOT and BLAST_WASM_CXX_INCLUDE, or install \
                 `wasi-libc` + `libc++-18-dev-wasm32` (Debian/Ubuntu)."
            );
        }

        // Tell rustc where to find `libc++.a` / `libc++abi.a` when the
        // final link step resolves the `-lc++` the `cc` crate injects
        // automatically for C++ source units. wasm32-unknown-unknown has
        // no libc++ in the rust sysroot, so we point it at the libcxx
        // shipped by `libc++-*-dev-wasm32`.
        //
        // BLAST_WASM_CXX_LIB_DIR lets users override for non-Debian
        // layouts; otherwise we probe a few well-known locations.
        let cxx_lib_dir = env::var("BLAST_WASM_CXX_LIB_DIR")
            .ok()
            .or_else(|| probe_libcxx_wasm_lib().map(|p| p.display().to_string()));
        if let Some(dir) = &cxx_lib_dir {
            println!("cargo:rerun-if-env-changed=BLAST_WASM_CXX_LIB_DIR");
            println!("cargo:rustc-link-search=native={dir}");
            // `cc` already passes `-lc++` on wasm32, but we also need
            // `c++abi` (exception unwinding stubs) even with -fno-exceptions.
            println!("cargo:rustc-link-lib=static=c++abi");
        } else {
            println!(
                "cargo:warning=blast-stress-solver: wasm libc++.a not found — \
                 set BLAST_WASM_CXX_LIB_DIR or install `libc++-18-dev-wasm32`."
            );
        }

        // We do NOT link wasi-libc.  Doing so would pull
        // `wasi_snapshot_preview1` imports into the final wasm, which
        // in turn causes wasm-bindgen to wrap every export in a
        // `command_export` shim that calls `__wasm_call_dtors →
        // __funcs_on_exit → __stdio_exit`.  Those walk wasi-libc's
        // atexit table, which is never initialised when we load the
        // module as a library, and every `__wbindgen_malloc` call
        // traps.
        //
        // Instead, we provide pure-Rust stubs for every libc symbol
        // that the Blast C++ backend references, in
        // `src/wasm_runtime_shims.rs`.  `malloc`/`free` forward to
        // Rust's global allocator; stdio is no-op; string / memory
        // helpers are pure Rust.  The final wasm has zero `env`
        // imports and zero `wasi_snapshot_preview1` imports, so
        // wasm-bindgen emits a normal library module.

        // Suppress a cascade of warnings from the NV headers that assume a
        // desktop-class compiler environment.
        build.flag_if_supported("-Wno-unknown-pragmas");
        build.flag_if_supported("-Wno-unused-parameter");
        build.flag_if_supported("-Wno-unused-variable");
        build.flag_if_supported("-Wno-deprecated-declarations");
    }
    // Silence noise about CARGO_CFG_* and target_os so incremental builds
    // know to rerun when the triple changes.
    let _ = &target_os;

    for src in bridge_sources.iter().chain(blast_sources.iter()) {
        build.file(src);
    }

    // Include paths
    build.include(blast.join("include"));
    build.include(blast.join("include/globals"));
    build.include(blast.join("include/lowlevel"));
    build.include(blast.join("include/extensions/stress"));
    build.include(blast.join("include/shared/NvFoundation"));
    build.include(blast.join("source/shared"));
    build.include(blast.join("source/shared/stress_solver"));
    build.include(blast.join("source/sdk/common"));
    build.include(blast.join("source/sdk/lowlevel"));
    build.include(blast.join("source/shared/NsFoundation/include"));
    // The FFI dir for stress_bridge.h
    build.include(&ffi_dir);

    // C++17
    build.flag_if_supported("-std=c++17");
    build.define("NDEBUG", None);
    build.flag_if_supported("-fno-exceptions");
    build.flag_if_supported("-fvisibility=hidden");

    // --- SIMD selection ------------------------------------------------------
    //
    // The Blast stress solver ships a hand-written AVX/FMA implementation of the
    // CGNR inner loop (the `AngLin6`/coupling matrix-vector products that dominate
    // `solve()`), selected at RUNTIME by a CPUID device query with a scalar
    // fallback (see `stress.cpp` `s_use_simd`). The upstream build here force-
    // disabled it on every target, leaving the solver scalar-only.
    //
    // We enable the SIMD path on x86/x86_64 native targets: the `__m256`/FMA
    // intrinsics exist there and AVX is universal on x86_64 (Sandy Bridge, 2011+),
    // and the runtime device query still picks the scalar kernels if a CPU somehow
    // lacks AVX/FMA. wasm and non-x86 arches stay scalar (the intrinsics don't
    // exist). The `force-scalar-solver` feature (or `BLAST_FORCE_SCALAR=1`) forces
    // the scalar path everywhere — an escape hatch for reproducibility or odd CPUs.
    println!("cargo:rerun-if-env-changed=BLAST_FORCE_SCALAR");
    let x86 = matches!(target_arch.as_str(), "x86" | "x86_64");
    let force_scalar = is_wasm
        || !x86
        || env::var_os("CARGO_FEATURE_FORCE_SCALAR_SOLVER").is_some()
        || env::var_os("BLAST_FORCE_SCALAR").is_some();

    if force_scalar {
        build.define("STRESS_SOLVER_FORCE_SCALAR", None);
        build.define("STRESS_SOLVER_NO_SIMD", None);
        build.define("STRESS_SOLVER_NO_DEVICE_QUERY", None);
    } else if build.get_compiler().is_like_msvc() {
        // MSVC: /arch:AVX2 enables 256-bit AVX + FMA3 intrinsics.
        build.flag("/arch:AVX2");
    } else {
        // GCC/Clang: AVX (256-bit) + FMA3 for the `_mm256_*`/`_mm256_fmadd_ps` ops.
        build.flag("-mavx");
        build.flag("-mfma");
    }

    build.compile("blast_stress_solver_ffi");
}

/// Probe a few well-known locations for a wasi sysroot. Returns the first
/// one that looks valid (contains `include/wasi/api.h`).
fn probe_wasi_sysroot() -> Option<PathBuf> {
    let candidates = [
        "/usr/share/wasi-sysroot",
        "/opt/wasi-sdk/share/wasi-sysroot",
        "/usr",
    ];
    for c in candidates.iter() {
        let p = PathBuf::from(c);
        if p.join("include/wasm32-wasi/wasi/api.h").exists()
            || p.join("include/wasi/api.h").exists()
        {
            return Some(p);
        }
    }
    None
}

/// Probe a few well-known locations for a directory containing
/// `libc++.a` built for wasm32. Matches the layout shipped by
/// Ubuntu's `libc++-*-dev-wasm32` package.
fn probe_libcxx_wasm_lib() -> Option<PathBuf> {
    let candidates = [
        "/usr/lib/llvm-18/lib/wasm32-wasi",
        "/usr/lib/llvm-19/lib/wasm32-wasi",
        "/usr/lib/llvm-20/lib/wasm32-wasi",
        "/usr/lib/llvm-17/lib/wasm32-wasi",
        "/usr/lib/llvm-16/lib/wasm32-wasi",
        "/usr/lib/wasm32-wasi",
        "/opt/wasi-sdk/share/wasi-sysroot/lib/wasm32-wasi",
    ];
    for c in candidates.iter() {
        let p = PathBuf::from(c);
        if p.join("libc++.a").exists() {
            return Some(p);
        }
    }
    None
}

/// Probe a few well-known locations for a wasm-compatible libc++ header
/// tree. Matches the layout shipped by Ubuntu's `libc++-*-dev-wasm32`.
fn probe_libcxx_wasm_include() -> Option<PathBuf> {
    let candidates = [
        "/usr/lib/llvm-18/include/wasm32-wasi/c++/v1",
        "/usr/lib/llvm-19/include/wasm32-wasi/c++/v1",
        "/usr/lib/llvm-20/include/wasm32-wasi/c++/v1",
        "/usr/lib/llvm-17/include/wasm32-wasi/c++/v1",
        "/usr/lib/llvm-16/include/wasm32-wasi/c++/v1",
        "/usr/include/wasm32-wasi/c++/v1",
        "/opt/wasi-sdk/share/wasi-sysroot/include/c++/v1",
    ];
    for c in candidates.iter() {
        let p = PathBuf::from(c);
        if p.join("new").exists() {
            return Some(p);
        }
    }
    None
}

/// Compile the PhysX backend and link the PhysX SDK.
///
/// Kept behind the `physx` feature so the crate's default build needs nothing
/// but a C++ toolchain. `PHYSX_ROOT` overrides the SDK location; a missing SDK
/// fails with the path it looked for rather than a wall of linker errors.
#[cfg(feature = "physx")]
fn build_physx_backend(blast: &Path) {
    const DEFAULT_PHYSX_ROOT: &str = "/root/PhysX/physx/install/linux-clang/PhysX";
    let root = PathBuf::from(
        env::var_os("PHYSX_ROOT").unwrap_or_else(|| DEFAULT_PHYSX_ROOT.into()),
    );
    let include = root.join("include");
    assert!(
        include.join("PxPhysicsAPI.h").is_file(),
        "PhysX SDK not found: expected {} (set PHYSX_ROOT)",
        include.display()
    );
    let src = blast.join("physx_backend/physx_backend.cpp");
    assert!(src.is_file(), "missing {}", src.display());
    println!("cargo:rerun-if-env-changed=PHYSX_ROOT");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", blast.join("physx_backend/physx_backend.h").display());

    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .file(&src)
        .include(blast.join("physx_backend"))
        .include(&include)
        .define("NDEBUG", None)
        .define("PX_PHYSX_STATIC_LIB", None)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-variable");
    b.compile("blast_physx_backend");

    // Static archives are order sensitive: dependents first.
    let libdir = root.join("bin/linux.x86_64/release");
    assert!(libdir.is_dir(), "PhysX libs not found at {}", libdir.display());
    println!("cargo:rustc-link-search=native={}", libdir.display());
    for lib in [
        "PhysXExtensions_static_64",
        "PhysX_static_64",
        "PhysXPvdSDK_static_64",
        "PhysXCooking_static_64",
        "PhysXCommon_static_64",
        "PhysXFoundation_static_64",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // PhysX dlopens libPhysXGpu_64.so at CUDA-context creation. Without an
    // rpath the GPU scene fails to construct and the backend correctly reports
    // "no GPU" -- which reads as a missing GPU rather than a missing link path.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");
}

// Silence dead_code on native builds where `Path` is unused by the wasm
// probe helpers.
#[allow(dead_code)]
fn _assert_path_in_scope(_: &Path) {}
