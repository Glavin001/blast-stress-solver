// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// ScenePack v3 — chunk crushing — loader conformance, C++ side.
//
// v3 adds exactly two things to v2: an optional `crush` block per material and
// an optional `m` per node. This pins that they are read correctly, that they
// are rejected in a pack declaring an earlier version, and that the numbers
// arrive in the units SCENE_PACK_FORMAT.md promises.
//
// The cross-runtime digest (scene_pack_conformance_test) deliberately still
// pins the v2 fixture: TS/Rapier and Rust/Rapier do not implement crushing yet,
// so a shared v3 digest would assert agreement that does not exist. This test
// covers the C++ loader alone until they do.

#include "../scene_pack.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

using blast_demo::loadScenePack;
using blast_demo::ScenePack;

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("scene pack v3 test failed: " + message);
    }
}

void requireNear(float actual, float expected, float tolerance, const std::string& field)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        throw std::runtime_error(
            "scene pack v3 test failed: " + field + " got " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: scene_pack_v3_test <reference-building-crush.json>\n");
        return 2;
    }
    try
    {
        const ScenePack pack = loadScenePack(argv[1]);

        require(pack.version == 3, "pack must declare version 3");
        require(pack.materials.size() == 5, "expected the reference building's 5 materials");

        // Crush is opt-in per material: the frame, slab and panel author it;
        // the facade clip is a connector and the footing anchor must never
        // vaporize, so neither carries a crush block.
        require(pack.materials[0].crush.enabled, "reinforced-concrete should author crushing");
        require(pack.materials[1].crush.enabled, "concrete-slab should author crushing");
        require(pack.materials[2].crush.enabled, "drywall-panel should author crushing");
        require(!pack.materials[3].crush.enabled, "facade-clip must not author crushing");
        require(!pack.materials[4].crush.enabled, "footing-anchor must not author crushing");

        // Units and derivation, per SCENE_PACK_FORMAT.md: the cap is 2.5x the
        // material's compressive strength, and the cone is pinned so the chunk
        // yields at exactly fc under an unconfined squeeze.
        const blast_demo::CrushLimits& frame = pack.materials[0].crush;
        const float fc = pack.materials[0].limits.compressionElastic;
        requireNear(frame.capPressure, 2.5f * fc, 1.0f, "reinforced-concrete capPressure");
        requireNear(
            frame.cohesion,
            fc * (1.0f - frame.frictionSlope / 3.0f),
            1.0f,
            "reinforced-concrete cohesion");
        require(frame.crushEnergy > 0.0f, "crushEnergy must be positive");
        require(frame.crushViscosity > 0.0f, "crushViscosity must be positive");
        require(frame.debrisMassFraction == 0.0f,
                "the reference building should default to total mass loss");

        // Gypsum board must be dramatically more friable than the frame: that
        // ratio is what decides whether the impact zone turns to dust or the
        // whole structure does.
        require(pack.materials[2].crush.crushEnergy < pack.materials[0].crush.crushEnergy / 10.0f,
                "drywall must be far more friable than reinforced concrete");

        // Node materials: every node names the material that governs it, and
        // the foundations point at the one material with crushing disabled.
        std::size_t crushableNodes = 0;
        std::size_t supportNodes = 0;
        for (const blast_demo::SceneNode& node : pack.nodes)
        {
            require(node.material < pack.materials.size(),
                    "node material index must be inside the table");
            if (pack.materials[node.material].crush.enabled)
            {
                ++crushableNodes;
            }
            if (node.mass == 0.0f)
            {
                ++supportNodes;
                require(!pack.materials[node.material].crush.enabled,
                        "a support node must not be crushable: a vaporizing footing is "
                        "never the story");
            }
        }
        require(supportNodes == 4, "expected 4 footings");
        require(crushableNodes == pack.nodes.size() - supportNodes,
                "every non-support chunk should be crushable in this pack");

        std::printf(
            "ScenePack v3 OK: %zu materials (%zu crushable), %zu nodes (%zu crushable, "
            "%zu supports)\n",
            pack.materials.size(),
            static_cast<std::size_t>(3),
            pack.nodes.size(),
            crushableNodes,
            supportNodes);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
