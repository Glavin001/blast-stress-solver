cmake_minimum_required(VERSION 3.20)
project(blast_stress_physx_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
option(BLAST_ENABLE_CUDA_STRESS "Build the CUDA stress solver prototype" ON)
if(BLAST_ENABLE_CUDA_STRESS)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else()
        message(WARNING "CUDA compiler unavailable; CUDA stress solver disabled")
        set(BLAST_ENABLE_CUDA_STRESS OFF)
    endif()
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BLAST_ROOT "${REPO_ROOT}/blast" CACHE PATH "Blast source root")
set(PHYSX_ROOT "/root/PhysX/physx" CACHE PATH "Built PhysX SDK root")
set(PHYSX_CONFIGURATION "release" CACHE STRING "PhysX binary configuration")
set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/linux.x86_64/${PHYSX_CONFIGURATION}" CACHE PATH
    "PhysX library directory")

if(NOT EXISTS "${PHYSX_ROOT}/include/PxPhysicsAPI.h")
    message(FATAL_ERROR "PHYSX_ROOT does not contain include/PxPhysicsAPI.h: ${PHYSX_ROOT}")
endif()

function(require_physx_library output name)
    find_library(${output}
        NAMES "${name}"
        PATHS "${PHYSX_LIB_DIR}"
        NO_DEFAULT_PATH
        REQUIRED)
    set(${output} "${${output}}" PARENT_SCOPE)
endfunction()

require_physx_library(PHYSX_EXTENSIONS PhysXExtensions_static_64)
require_physx_library(PHYSX_CORE PhysX_static_64)
require_physx_library(PHYSX_PVD PhysXPvdSDK_static_64)
require_physx_library(PHYSX_COOKING PhysXCooking_static_64)
require_physx_library(PHYSX_COMMON PhysXCommon_static_64)
require_physx_library(PHYSX_FOUNDATION PhysXFoundation_static_64)
require_physx_library(PHYSX_CUDA_MANAGER PhysXCudaContextManager_static_64)
find_library(PVD_RUNTIME
    NAMES PVDRuntime_64
    PATHS "${PHYSX_LIB_DIR}"
    NO_DEFAULT_PATH)

set(BLAST_CORE_SOURCES
    "${BLAST_ROOT}/rust_stress_example/ffi/ext_stress_bridge.cpp"
    "${BLAST_ROOT}/source/shared/stress_solver/stress.cpp"
    "${BLAST_ROOT}/source/sdk/common/NvBlastAssert.cpp"
    "${BLAST_ROOT}/source/sdk/common/NvBlastAtomic.cpp"
    "${BLAST_ROOT}/source/sdk/common/NvBlastTime.cpp"
    "${BLAST_ROOT}/source/sdk/common/NvBlastTimers.cpp"
    "${BLAST_ROOT}/source/sdk/globals/NvBlastGlobals.cpp"
    "${BLAST_ROOT}/source/sdk/globals/NvBlastInternalProfiler.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastAsset.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastAssetHelper.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastFamily.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastFamilyGraph.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastActor.cpp"
    "${BLAST_ROOT}/source/sdk/lowlevel/NvBlastActorSerializationBlock.cpp"
    "${BLAST_ROOT}/source/sdk/extensions/stress/NvBlastExtStressSolver.cpp")

add_library(blast_stress_core STATIC ${BLAST_CORE_SOURCES})
target_include_directories(blast_stress_core PUBLIC
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/globals"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/extensions/stress"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/source/shared"
    "${BLAST_ROOT}/source/shared/stress_solver"
    "${BLAST_ROOT}/source/sdk/common"
    "${BLAST_ROOT}/source/sdk/lowlevel"
    "${BLAST_ROOT}/source/shared/NsFoundation/include"
    "${BLAST_ROOT}/rust_stress_example/ffi")
target_compile_definitions(blast_stress_core PRIVATE NDEBUG)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(blast_stress_core PRIVATE -mavx -mfma)
else()
    target_compile_definitions(blast_stress_core PRIVATE
        STRESS_SOLVER_FORCE_SCALAR
        STRESS_SOLVER_NO_SIMD
        STRESS_SOLVER_NO_DEVICE_QUERY)
endif()

file(GLOB STRESS_PHYSX_SOURCES CONFIGURE_DEPENDS
    "${BLAST_ROOT}/source/sdk/extensions/stressphysx/*.cpp")
if(NOT STRESS_PHYSX_SOURCES)
    message(FATAL_ERROR "PhysX adapter sources are missing under blast/source/sdk/extensions/stressphysx")
endif()

add_library(NvBlastExtStressPhysX STATIC ${STRESS_PHYSX_SOURCES})
target_include_directories(NvBlastExtStressPhysX PUBLIC
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${BLAST_ROOT}/rust_stress_example/ffi"
    "${PHYSX_ROOT}/include")
target_compile_definitions(NvBlastExtStressPhysX PUBLIC PX_PHYSX_STATIC_LIB NDEBUG)
target_link_libraries(NvBlastExtStressPhysX PUBLIC blast_stress_core)

if(BLAST_ENABLE_CUDA_STRESS)
    find_package(CUDAToolkit REQUIRED)
    add_library(NvBlastExtStressGpu STATIC
        "${BLAST_ROOT}/source/sdk/extensions/stressgpu/NvBlastExtStressGpu.cu")
    target_include_directories(NvBlastExtStressGpu PUBLIC
        "${BLAST_ROOT}/include/extensions/stressgpu")
    set_target_properties(NvBlastExtStressGpu PROPERTIES
        CUDA_ARCHITECTURES native
        POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(blast_stress_core PRIVATE
        NVBLAST_ENABLE_CUDA_STRESS)
    target_link_libraries(NvBlastExtStressGpu PUBLIC
        CUDA::cudart
        CUDA::cuda_driver)
    target_link_libraries(blast_stress_core PUBLIC NvBlastExtStressGpu)
endif()

set(PHYSX_LINK_LIBS
    "-Wl,--start-group"
    ${PHYSX_EXTENSIONS}
    ${PHYSX_CUDA_MANAGER}
    ${PHYSX_CORE}
    ${PHYSX_PVD}
    ${PHYSX_COOKING}
    ${PHYSX_COMMON}
    ${PHYSX_FOUNDATION}
    "-Wl,--end-group"
    ${PVD_RUNTIME}
    cuda
    dl
    pthread
    rt)

add_executable(blast_stress_demo
    mini_city_main.cpp
    physx_scene.cpp
    scene_pack.cpp
    state_writer.cpp)
target_include_directories(blast_stress_demo PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(blast_stress_demo PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(blast_stress_demo PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(blast_stress_demo PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})

add_executable(blast_direct_gpu_p0
    direct_gpu_p0.cpp
    physx_scene.cpp)
target_include_directories(blast_direct_gpu_p0 PRIVATE
    "${PHYSX_ROOT}/include")
target_compile_definitions(blast_direct_gpu_p0 PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(blast_direct_gpu_p0 PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(blast_direct_gpu_p0 PRIVATE
    ${PHYSX_LINK_LIBS})

enable_testing()
add_test(NAME blast_stress_physx_cpu_contract
    COMMAND blast_stress_demo --self-test --physics cpu)
add_test(NAME blast_stress_physx_resim_penetration
    COMMAND blast_stress_demo --self-test --physics cpu
        --resim-passes 1 --resim-assert penetrate)
add_test(NAME blast_stress_physx_resim_baseline
    COMMAND blast_stress_demo --self-test --physics cpu
        --resim-passes 0 --resim-assert deflect)

add_executable(resim_snapshot_test
    tests/resim_snapshot_test.cpp
    physx_scene.cpp)
target_include_directories(resim_snapshot_test PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(resim_snapshot_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(resim_snapshot_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(resim_snapshot_test PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_physx_resim_snapshot COMMAND resim_snapshot_test)

add_executable(load_path_test
    tests/load_path_test.cpp
    physx_scene.cpp)
target_include_directories(load_path_test PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(load_path_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(load_path_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(load_path_test PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_load_path COMMAND load_path_test)

add_executable(resim_punchthrough_test
    tests/resim_punchthrough_test.cpp
    physx_scene.cpp)
target_include_directories(resim_punchthrough_test PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(resim_punchthrough_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(resim_punchthrough_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(resim_punchthrough_test PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_physx_resim_punchthrough COMMAND resim_punchthrough_test)

add_executable(resim_scoped_cascade_test
    tests/resim_scoped_cascade_test.cpp
    physx_scene.cpp)
target_include_directories(resim_scoped_cascade_test PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(resim_scoped_cascade_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(resim_scoped_cascade_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(resim_scoped_cascade_test PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_physx_resim_scoped_cascade COMMAND resim_scoped_cascade_test)

add_executable(direct_gpu_resim_test
    tests/direct_gpu_resim_test.cpp
    physx_scene.cpp)
target_include_directories(direct_gpu_resim_test PRIVATE
    "${BLAST_ROOT}/include"
    "${BLAST_ROOT}/include/lowlevel"
    "${BLAST_ROOT}/include/shared/NvFoundation"
    "${BLAST_ROOT}/include/extensions/stressphysx"
    "${PHYSX_ROOT}/include")
target_compile_definitions(direct_gpu_resim_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(direct_gpu_resim_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}"
    "-Wl,--gc-sections")
target_link_libraries(direct_gpu_resim_test PRIVATE
    NvBlastExtStressPhysX
    ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_physx_direct_gpu_resim COMMAND direct_gpu_resim_test)

add_executable(scene_pack_test
    tests/scene_pack_test.cpp
    scene_pack.cpp)
target_include_directories(scene_pack_test PRIVATE "${PHYSX_ROOT}/include")
target_compile_definitions(scene_pack_test PRIVATE PX_PHYSX_STATIC_LIB)
add_executable(scene_pack_conformance_test
    tests/scene_pack_conformance_test.cpp
    scene_pack.cpp)
target_include_directories(scene_pack_conformance_test PRIVATE
    "${PHYSX_ROOT}/include")
target_compile_definitions(scene_pack_conformance_test PRIVATE PX_PHYSX_STATIC_LIB NDEBUG)
target_link_options(scene_pack_conformance_test PRIVATE
    "-Wl,-rpath,${PHYSX_LIB_DIR}")
target_link_libraries(scene_pack_conformance_test PRIVATE ${PHYSX_LINK_LIBS})
add_test(NAME blast_stress_scene_pack_conformance
    COMMAND scene_pack_conformance_test
        "${REPO_ROOT}/blast/blast-stress-solver/assets/conformance/structure-conformance-v2.json")

add_test(NAME blast_stress_scene_pack
    COMMAND scene_pack_test
        "${BLAST_ROOT}/blast-stress-demo-rs/assets/scenes/fractured-tower.json")

if(BLAST_ENABLE_CUDA_STRESS)
    add_executable(gpu_stress_test tests/gpu_stress_test.cpp)
    target_include_directories(gpu_stress_test PRIVATE
        "${BLAST_ROOT}/include/extensions/stressgpu"
        "${BLAST_ROOT}/source/shared"
        "${BLAST_ROOT}/source/shared/stress_solver"
        "${BLAST_ROOT}/include/shared/NvFoundation")
    target_compile_definitions(gpu_stress_test PRIVATE
        STRESS_SOLVER_FORCE_SCALAR
        STRESS_SOLVER_NO_SIMD
        STRESS_SOLVER_NO_DEVICE_QUERY)
    target_link_libraries(gpu_stress_test PRIVATE
        NvBlastExtStressGpu
        blast_stress_core)
    add_test(NAME blast_stress_gpu_equivalence COMMAND gpu_stress_test)

    add_executable(gpu_stress_benchmark tests/gpu_stress_benchmark.cpp)
    target_include_directories(gpu_stress_benchmark PRIVATE
        "${BLAST_ROOT}/include/extensions/stressgpu"
        "${BLAST_ROOT}/source/shared"
        "${BLAST_ROOT}/source/shared/stress_solver"
        "${BLAST_ROOT}/include/shared/NvFoundation")
    target_link_libraries(gpu_stress_benchmark PRIVATE
        NvBlastExtStressGpu
        blast_stress_core)
endif()
