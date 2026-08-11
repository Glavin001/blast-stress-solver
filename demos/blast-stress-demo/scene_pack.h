// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <PxPhysicsAPI.h>

namespace blast_demo
{

enum class SceneColliderKind
{
    Cuboid,
    ConvexHull
};

struct SceneCollider
{
    SceneColliderKind kind{SceneColliderKind::Cuboid};
    physx::PxVec3 halfExtents{0.5f};
    std::vector<physx::PxVec3> points;
    std::uint32_t sourcePointCount{0};
};

struct SceneNode
{
    physx::PxVec3 centroid{0.0f};
    float mass{0.0f};
    float volume{0.0f};
    physx::PxVec3 visualHalfExtents{0.5f};
    SceneCollider collider;
};

/**
 * Optional real render geometry for one node (`scenario.nodeMeshes[i]` in the
 * pack JSON) — e.g. a Voronoi-fractured shard's actual irregular hull, so it
 * draws as its true shape instead of its AABB box. Positions/normals are
 * parallel per-vertex arrays, already centroid-relative (the same convention
 * as `SceneCollider::points` and `SceneNode::visualHalfExtents`).
 *
 * `present` is false when this node has no mesh entry (absent, or `null` in
 * the array) — the normal case for structural nodes (columns/slabs/
 * foundation) that render as their box collider. A pack may mix meshed and
 * box-only nodes freely.
 */
struct SceneMesh
{
    bool present{false};
    std::vector<physx::PxVec3> positions;
    std::vector<physx::PxVec3> normals;
    std::vector<std::uint32_t> indices;
};

struct SceneBond
{
    std::uint32_t node0{0};
    std::uint32_t node1{0};
    physx::PxVec3 centroid{0.0f};
    physx::PxVec3 normal{0.0f, 1.0f, 0.0f};
    // Geometry: the real contact patch (m^2), and the bond's damage pool.
    // Never a strength knob — strength is `material`. See SCENE_PACK_FORMAT.md.
    float area{1.0f};
    // Index into ScenePack::materials (JSON `m`; absent means 0).
    std::uint32_t material{0};
};

struct StressLimits
{
    float compressionElastic{1.0e6f};
    float compressionFatal{2.0e6f};
    float tensionElastic{1.0e6f};
    float tensionFatal{2.0e6f};
    float shearElastic{1.0e6f};
    float shearFatal{2.0e6f};
};

/**
 * One entry of the pack's material table. `name` is author-defined and exists
 * for reports and debugging — there is no material enum and the library ships
 * no material library; see SCENE_PACK_FORMAT.md.
 *
 * Ductility is the width of the (fatal - elastic) band, independent of raw
 * strength: wide = yields over many frames, narrow = snaps.
 */
struct SceneMaterial
{
    std::string name;
    StressLimits limits;
};

struct ScenePack
{
    std::string title;
    std::vector<SceneNode> nodes;
    std::vector<SceneBond> bonds;
    // Optional authored node roles ("foundation", "column", "slab", "infill", ...),
    // parallel to `nodes`. Empty when the pack omits `scenario.nodeTypes`. Used to
    // label joint classes in the load-path safety-factor report.
    std::vector<std::string> nodeTypes;
    // Optional real render geometry, parallel to `nodes` and always sized to
    // match it (padded with `present=false` entries) once a pack is loaded, so
    // callers can index nodeMeshes[i] unconditionally. See SceneMesh.
    std::vector<SceneMesh> nodeMeshes;
    float gravity{-9.81f};
    float projectileRadius{0.6f};
    float projectileMass{1500.0f};
    float projectileSpeed{60.0f};
    float projectileTtlSeconds{8.0f};
    float friction{0.25f};
    float restitution{0.0f};
    // Contact impulse -> stress force transfer. 1.0 is the physically correct
    // value: the adapter divides the solved impulse by dt to get a force
    // (NvBlastExtStressPhysX.cpp, consumeContacts). Values above 1 are gain
    // that has to be cancelled elsewhere.
    float contactForceScale{1.0f};
    // Material table. Always >= 1 entry: a v1 pack synthesizes a single entry
    // from `solver.limits` (or the placeholder default) so every runtime sees
    // the same shape regardless of pack version.
    std::vector<SceneMaterial> materials;
    // Convenience alias for materials[0].limits, the structure default.
    StressLimits stressLimits;
    // False when the pack stated no material at all (v1 without solver.limits)
    // and stressLimits holds placeholder values.
    bool stressLimitsAuthored{false};
    // Pack schema version as loaded (1 or 2).
    std::uint32_t version{1};
};

ScenePack loadScenePack(const std::string& path);

} // namespace blast_demo
