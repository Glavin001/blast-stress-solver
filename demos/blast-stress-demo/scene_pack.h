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

struct SceneBond
{
    std::uint32_t node0{0};
    std::uint32_t node1{0};
    physx::PxVec3 centroid{0.0f};
    physx::PxVec3 normal{0.0f, 1.0f, 0.0f};
    float area{1.0f};
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

struct ScenePack
{
    std::string title;
    std::vector<SceneNode> nodes;
    std::vector<SceneBond> bonds;
    // Optional authored node roles ("foundation", "column", "slab", "infill", ...),
    // parallel to `nodes`. Empty when the pack omits `scenario.nodeTypes`. Used to
    // label joint classes in the load-path safety-factor report.
    std::vector<std::string> nodeTypes;
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
    StressLimits stressLimits;
    // False when the pack omitted defaults.solver.limits and stressLimits holds
    // placeholder values rather than an authored material.
    bool stressLimitsAuthored{false};
};

ScenePack loadScenePack(const std::string& path);

} // namespace blast_demo
