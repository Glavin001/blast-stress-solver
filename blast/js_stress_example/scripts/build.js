import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = resolve(__dirname, '..');
const blastRoot = resolve(projectRoot, '..');
const distDir = resolve(projectRoot, 'dist');
const jsFfiDir = resolve(projectRoot, 'ffi');

mkdirSync(distDir, { recursive: true });

const ffiDir = resolve(blastRoot, 'rust_stress_example/ffi');
const solverDir = resolve(blastRoot, 'source/shared/stress_solver');
const sharedDir = resolve(blastRoot, 'source/shared');
const sharedNsFoundationIncludeDir = resolve(sharedDir, 'NsFoundation', 'include');
const includeDir = resolve(blastRoot, 'include');
const includeGlobalsDir = resolve(includeDir, 'globals');
const includeSharedDir = resolve(includeDir, 'shared');
const includeLowLevelDir = resolve(includeDir, 'lowlevel');
const includeExtensionsDir = resolve(includeDir, 'extensions');
const includeStressExtDir = resolve(includeExtensionsDir, 'stress');
const includeAuthoringDir = resolve(includeExtensionsDir, 'authoring');
const includeAuthoringCommonDir = resolve(includeExtensionsDir, 'authoringCommon');
const includeAssetUtilsDir = resolve(includeExtensionsDir, 'assetutils');
const foundationDir = resolve(includeSharedDir, 'NvFoundation');
const sdkCommonDir = resolve(blastRoot, 'source/sdk/common');
const sdkGlobalsDir = resolve(blastRoot, 'source/sdk/globals');
const sdkLowLevelDir = resolve(blastRoot, 'source/sdk/lowlevel');
const sdkStressDir = resolve(blastRoot, 'source/sdk/extensions/stress');
const sdkAuthoringDir = resolve(blastRoot, 'source/sdk/extensions/authoring');
const sdkAuthoringCommonDir = resolve(blastRoot, 'source/sdk/extensions/authoringCommon');

const exportedFunctions = [
  '_stress_processor_create',
  '_stress_processor_destroy',
  '_stress_processor_node_count',
  '_stress_processor_bond_count',
  '_stress_processor_solve',
  '_stress_processor_remove_bond',
  '_stress_processor_get_node_desc',
  '_stress_processor_get_bond_desc',
  '_stress_processor_using_simd',
  '_stress_sizeof_stress_vec3',
  '_stress_sizeof_node_desc',
  '_stress_sizeof_bond_desc',
  '_stress_sizeof_velocity',
  '_stress_sizeof_impulse',
  '_stress_sizeof_data_params',
  '_stress_sizeof_solver_params',
  '_stress_sizeof_error_sq',
  '_ext_stress_solver_create',
  '_ext_stress_solver_destroy',
  '_ext_stress_solver_set_settings',
  '_ext_stress_solver_graph_node_count',
  '_ext_stress_solver_bond_count',
  '_ext_stress_solver_reset',
  '_ext_stress_solver_add_force',
  '_ext_stress_solver_add_all_forces',
  '_ext_stress_solver_add_gravity',
  '_ext_stress_solver_add_actor_gravity',
  '_ext_stress_solver_add_centrifugal_acceleration',
  '_ext_stress_solver_add_all_actor_gravity',
  '_ext_stress_solver_update',
  '_ext_stress_solver_overstressed_bond_count',
  '_ext_stress_solver_fill_debug_render',
  '_ext_stress_solver_generate_fracture_commands',
  '_ext_stress_solver_actor_count',
  '_ext_stress_solver_collect_actors',
  '_ext_stress_solver_generate_fracture_commands_per_actor',
  '_ext_stress_solver_apply_fracture_commands',
  '_ext_stress_solver_get_excess_forces',
  '_ext_stress_solver_get_linear_error',
  '_ext_stress_solver_get_angular_error',
  '_ext_stress_solver_converged',
  '_ext_stress_solver_island_count',
  '_ext_stress_solver_set_island_aware',
  '_ext_stress_solver_get_island_aware',
  '_ext_stress_solver_set_skip_settled',
  '_ext_stress_solver_get_skip_settled',
  '_ext_stress_solver_islands_skipped',
  '_ext_stress_solver_islands_total',
  '_ext_stress_sizeof_ext_node_desc',
  '_ext_stress_sizeof_ext_bond_desc',
  '_ext_stress_sizeof_ext_settings',
  '_ext_stress_sizeof_ext_debug_line',
  '_ext_stress_sizeof_ext_bond_fracture',
  '_ext_stress_sizeof_ext_fracture_commands',
  '_ext_stress_sizeof_actor',
  '_ext_stress_sizeof_actor_buffer',
  '_ext_stress_sizeof_ext_split_event',
  '_authoring_sizeof_ext_bond_desc',
  '_authoring_bonds_from_prefractured_triangles',
  '_authoring_free',
  '_malloc',
  '_free'
];

// Default-off for production: a debug WASM with assertions is 4x larger, slower
// to load, and runs significantly slower in the hot solver loop. Opt in with
// EMCC_ASSERTIONS=1 when bisecting a crash; default builds are stripped.
const enableAssertions = process.env.EMCC_ASSERTIONS === '1';
const enableProfiling = process.env.EMCC_PROFILING === '1';
// Opt-out for LTO: link-time optimization gives whole-program inlining across
// the 22 translation units below (notably inlining the CGNR kernels into
// ext_stress_solver_update), at the cost of ~20s extra link time. Disable with
// EMCC_NO_LTO=1 if iterating locally and only the JS bridge changed.
const enableLTO = process.env.EMCC_NO_LTO !== '1';
// Opt-in for the experimental simde-routed AVX path. Off by default — the
// scalar CGNR + clang autovec (`-O3 -msimd128`) is what every shipped tests
// has run against. Enable with `EMCC_USE_SIMD=1` to build the alternate path
// (anglin6.h / coupling.h / inertia.h `<__m128>` specializations compiled with
// `__m256` intrinsics rerouted to wasm-simd128 via simde + an FMA macro
// override).  Build only — selecting the SIMD path at runtime is the
// `s_use_simd` flag in stress.cpp, which is true when this define is set.
const enableSimd = process.env.EMCC_USE_SIMD === '1';
const simdeIncludeDir = resolve(blastRoot, 'deps/simde');

if (enableSimd && !existsSync(resolve(simdeIncludeDir, 'simde/x86/avx.h'))) {
  console.error(
    '[blast-stress-solver] EMCC_USE_SIMD=1 requires the simde submodule. ' +
      `Headers not found under ${simdeIncludeDir}. Run:\n` +
      '  git submodule update --init --recursive blast/deps/simde'
  );
  process.exit(1);
}

// EXPORTED_RUNTIME_METHODS only covers JS helpers — HEAPU8/HEAPU32/HEAPF32 are
// auto-attached to Module by emscripten's runtime when ALLOW_MEMORY_GROWTH=1
// (the TS bridge re-reads .HEAPU8.buffer after grow events), so listing them
// here is a no-op that emits a warning per build.
const exportedRuntimeMethods = [
  'cwrap',
  'ccall',
  'UTF8ToString'
];

const commonArgs = [
  resolve(ffiDir, 'stress_bridge.cpp'),
  resolve(ffiDir, 'ext_stress_bridge.cpp'),
  resolve(jsFfiDir, 'authoring_bridge.cpp'),
  resolve(solverDir, 'stress.cpp'),
  resolve(sdkStressDir, 'NvBlastExtStressSolver.cpp'),
  resolve(sdkAuthoringDir, 'NvBlastExtAuthoring.cpp'),
  resolve(sdkAuthoringDir, 'NvBlastExtAuthoringBondGeneratorImpl.cpp'),
  resolve(sdkAuthoringDir, 'NvBlastExtTriangleProcessor.cpp'),
  resolve(sdkAuthoringDir, 'NvBlastExtApexSharedParts.cpp'),
  resolve(sdkCommonDir, 'NvBlastAssert.cpp'),
  resolve(sdkCommonDir, 'NvBlastAtomic.cpp'),
  resolve(sdkCommonDir, 'NvBlastTime.cpp'),
  resolve(sdkCommonDir, 'NvBlastTimers.cpp'),
  resolve(sdkGlobalsDir, 'NvBlastGlobals.cpp'),
  resolve(sdkGlobalsDir, 'NvBlastInternalProfiler.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastActor.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastActorSerializationBlock.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastAsset.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastAssetHelper.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastFamily.cpp'),
  resolve(sdkLowLevelDir, 'NvBlastFamilyGraph.cpp'),
  '-I' + ffiDir,
  '-I' + solverDir,
  '-I' + sharedDir,
  '-I' + sharedNsFoundationIncludeDir,
  '-I' + includeDir,
  '-I' + includeGlobalsDir,
  '-I' + includeSharedDir,
  '-I' + includeLowLevelDir,
  '-I' + includeExtensionsDir,
  '-I' + includeStressExtDir,
  '-I' + includeAuthoringDir,
  '-I' + includeAuthoringCommonDir,
  '-I' + includeAssetUtilsDir,
  '-I' + foundationDir,
  '-I' + sdkCommonDir,
  '-I' + sdkGlobalsDir,
  '-I' + sdkLowLevelDir,
  '-I' + sdkStressDir,
  '-I' + sdkAuthoringDir,
  '-I' + sdkAuthoringCommonDir,
  ...(enableSimd
    ? [
        // SIMD path: compile the vendored __m256 AngLin6 kernels via simde →
        // wasm-simd128. STRESS_SOLVER_WASM_SIMD turns on the simde branch in
        // simd/simd.h (which redefines _mm{_256_,}fmadd_ps to v128 macros) and
        // flips StressProcessor::s_use_simd to true so the runtime picks the
        // CGNR_SIMD code path instead of CGNR_SISD.
        '-I' + simdeIncludeDir,
        '-DSTRESS_SOLVER_WASM_SIMD=1',
      ]
    : [
        // Scalar path (default): the __m256 specialization in anglin6.h /
        // coupling.h / inertia.h won't compile under emscripten (no native
        // __m256 / _mm_fmadd_ps), so gate it off and let the autovectorizer
        // do its job on the scalar AngLin6Ops<float> path.
        '-DSTRESS_SOLVER_FORCE_SCALAR=1',
        '-DSTRESS_SOLVER_NO_SIMD=1',
      ]),
  '-D__linux__=1',
  '-D__arm__=1',
  '-DCOMPILE_VECTOR_INTRINSICS=0',
  '-DNDEBUG=1',
  '-std=c++17',
  '-O3',
  // -msimd128 enables WASM SIMD auto-vectorization for the scalar math loops in
  // anglin6.h / inertia.h / coupling.h (the SISD AngLin6Ops path; the vendored
  // __m256 specializations are normally gated off behind STRESS_SOLVER_NO_SIMD
  // because emscripten 3.1.51 has no __m256/_mm_fmadd_ps).  When EMCC_USE_SIMD=1
  // we instead route them through simde, which expands them to v128 ops.
  // -msse4.2 widens the intrinsic pool clang can fuse loads/shuffles into.
  '-msimd128',
  '-msse4.2',
  // Whole-program / no-runtime-overhead settings. The C++ here doesn't throw,
  // doesn't use RTTI, doesn't longjmp, doesn't touch a filesystem, and only
  // uses fprintf(stderr, ...) for one warning (which emcc still satisfies
  // without FILESYSTEM via the JS console). Dropping each gives smaller WASM,
  // less JS glue, and (for -flto / -fno-exceptions) tighter codegen.
  '-fno-exceptions',
  '-fno-rtti',
  '-fno-math-errno',         // Don't set errno from <math.h>; safe — code never reads errno.
  '-fno-trapping-math',      // Assume FP ops don't trap; safe — no FE_* trap handling.
  '-fno-stack-protector',    // Stack canaries are a no-op on wasm32 anyway.
  '-fvisibility=hidden',     // Internal symbols don't reach the JS export table.
  '-sWASM=1',
  '-sMODULARIZE=1',
  '-sALLOW_MEMORY_GROWTH=1',
  '-sFILESYSTEM=0',          // No FS — saves ~30 KB of JS glue + runtime init.
  '-sSUPPORT_LONGJMP=0',     // No setjmp/longjmp in any TU.
  '-sNO_EXIT_RUNTIME=1',     // Module lives for the process; no atexit/_exit.
  '-sDISABLE_EXCEPTION_CATCHING=1',  // Pair with -fno-exceptions.
  '-sDYNAMIC_EXECUTION=0',   // No eval() / new Function() — smaller JS, CSP-safe.
  '-sSTACK_SIZE=1048576',    // 1 MiB stack (default is 64 KiB): provides headroom for the solver's heavy AngLin6 temporaries on large graphs.
  // Extra Binaryen passes layered on top of the default `wasm-opt -O3` run.
  // `--gufa` (Global Use Flow Analysis) is whole-program; this WASM is closed
  // (no dynamic linking) so its extra precision pays off. `--converge` then
  // re-runs the standard pipeline until a fixed point, catching simplifications
  // exposed by GUFA. Both add a few seconds of link time.
  '-sBINARYEN_EXTRA_PASSES=--gufa,--converge',
  `-sEXPORTED_FUNCTIONS=[${exportedFunctions.map((fn) => `"${fn}"`).join(',')}]`,
  `-sEXPORTED_RUNTIME_METHODS=[${exportedRuntimeMethods.map((name) => `"${name}"`).join(',')}]`
];

if (enableLTO) {
  // -flto applies to both compile (each .cpp -> bitcode) and link (whole-program
  // inlining + dead-code elimination across TUs). Required at both stages.
  commonArgs.push('-flto');
}

if (enableAssertions) {
  console.log('Building with assertions');
  commonArgs.push('-sASSERTIONS=1');
} else {
  commonArgs.push('-sASSERTIONS=0');
}

if (enableProfiling) {
  console.log('Building with profiling symbols');
  commonArgs.push('--profiling');
}

const builds = [
  {
    name: 'node-cjs',
    args: [
      ...commonArgs,
      '-sEXPORT_NAME=createStressModule',
      '-sENVIRONMENT=node',
      '-o',
      resolve(distDir, 'stress_solver.cjs')
    ]
  },
  {
    name: 'browser-esm',
    args: [
      ...commonArgs,
      '-sEXPORT_ES6=1',
      '-sENVIRONMENT=web,worker,node',
      '-sEXPORT_NAME=createStressModule',
      '-o',
      resolve(distDir, 'stress_solver.mjs')
    ]
  }
];

for (const build of builds) {
  const result = spawnSync('emcc', build.args, { stdio: 'inherit' });

  if (result.error) {
    console.error(`Failed ${build.name} build:`, result.error);
    process.exit(1);
  }

  if (result.status !== 0) {
    console.error(`Emscripten exited with code ${result.status} for ${build.name}`);
    process.exit(result.status ?? 1);
  }
}

const artifacts = ['stress_solver.cjs', 'stress_solver.mjs', 'stress_solver.wasm'];
for (const file of artifacts) {
  const src = resolve(distDir, file);
  const dst = resolve(projectRoot, file);
  copyFileSync(src, dst);
}
