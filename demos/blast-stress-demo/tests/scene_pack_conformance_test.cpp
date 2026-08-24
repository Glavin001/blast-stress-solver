// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Cross-runtime ScenePack conformance — C++ side.
//
// The ScenePack JSON is the contract that lets the same structure run under
// TS/Rapier, Rust/Rapier and C++/PhysX so their APIs, behavior and performance
// can be compared without the structure itself being a variable. That only
// holds if all three loaders interpret the file identically. They previously
// validated `version == 1` and each read a different subset of fields, so any
// divergence — a silently ignored field, an off-by-one index, a misread unit —
// was invisible.
//
// Each runtime loads the SAME fixture and computes the SAME digest, asserted
// against one golden file. If a loader drifts, that runtime's own suite fails.
//
// The digest pins interpretation of the ASSET, not simulation results: Rapier
// and PhysX legitimately produce different trajectories from identical input.
//
// See SCENE_PACK_FORMAT.md ("Conformance") and the sibling tests:
//   blast/blast-stress-solver/src/tests/scenePack.conformance.test.ts
//   blast/blast-stress-demo-rs/tests/scene_pack_conformance.rs

#include "../scene_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using blast_demo::ScenePack;
using blast_demo::loadScenePack;

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("scene pack conformance failed: " + message);
    }
}

template <typename T>
void requireEqual(const T& actual, const T& expected, const std::string& field)
{
    if (!(actual == expected))
    {
        std::ostringstream out;
        out << field << ": got " << actual << ", golden says " << expected;
        throw std::runtime_error("scene pack conformance failed: " + out.str());
    }
}

void requireNear(double actual, double expected, double tolerance, const std::string& field)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        std::ostringstream out;
        out << field << ": got " << actual << ", golden says " << expected;
        throw std::runtime_error("scene pack conformance failed: " + out.str());
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: scene_pack_conformance_test <fixture.json>\n");
        return 2;
    }

    try
    {
        const ScenePack pack = loadScenePack(argv[1]);

        // --- Golden digest, mirrored in the TS and Rust conformance tests ---
        requireEqual<std::uint32_t>(pack.version, 2u, "version");
        requireEqual<std::size_t>(pack.nodes.size(), 7u, "nodeCount");
        requireEqual<std::size_t>(pack.bonds.size(), 7u, "bondCount");
        requireEqual<std::size_t>(pack.materials.size(), 3u, "materialCount");

        const std::vector<std::string> expectedNames{
            "reinforced-concrete", "concrete", "drywall-track"};
        for (std::size_t i = 0; i < expectedNames.size(); ++i)
        {
            requireEqual(pack.materials[i].name, expectedNames[i],
                         "materialNames[" + std::to_string(i) + "]");
        }

        std::size_t supportNodes = 0;
        double totalMass = 0.0;
        for (const auto& node : pack.nodes)
        {
            if (node.mass == 0.0f) ++supportNodes;
            totalMass += node.mass;
        }
        requireEqual<std::size_t>(supportNodes, 2u, "supportNodeCount");
        requireNear(totalMass, 6000.0, 1e-3, "totalMassKg");

        double totalArea = 0.0;
        std::vector<std::size_t> bondsPerMaterial(pack.materials.size(), 0);
        std::map<std::string, std::size_t> bondsPerClass;
        for (const auto& bond : pack.bonds)
        {
            totalArea += bond.area;
            require(bond.material < bondsPerMaterial.size(), "bond material index in range");
            ++bondsPerMaterial[bond.material];
            std::string a = pack.nodeTypes[bond.node0];
            std::string b = pack.nodeTypes[bond.node1];
            if (b < a) std::swap(a, b);
            ++bondsPerClass[a + "~" + b];
        }
        requireNear(totalArea, 1.46, 1e-6, "totalBondAreaM2");

        const std::vector<std::size_t> expectedPerMaterial{2, 2, 3};
        for (std::size_t i = 0; i < expectedPerMaterial.size(); ++i)
        {
            requireEqual(bondsPerMaterial[i], expectedPerMaterial[i],
                         "bondsPerMaterial[" + std::to_string(i) + "]");
        }

        const std::map<std::string, std::size_t> expectedPerClass{
            {"column~foundation", 2}, {"column~infill", 2},
            {"column~slab", 2}, {"infill~infill", 1}};
        requireEqual(bondsPerClass.size(), expectedPerClass.size(), "jointClassCount");
        for (const auto& entry : expectedPerClass)
        {
            const auto found = bondsPerClass.find(entry.first);
            require(found != bondsPerClass.end(), "missing joint class " + entry.first);
            requireEqual(found->second, entry.second, "bondsPerJointClass[" + entry.first + "]");
        }

        requireNear(pack.gravity, -9.81, 1e-6, "gravity");
        requireNear(pack.contactForceScale, 1.0, 1e-6, "contactForceScale");

        // The material index must reach the right bonds, not just count right:
        // the facade clips are drywall-track, the footings the frame default.
        requireEqual(pack.materials[pack.bonds[0].material].name,
                     std::string("reinforced-concrete"), "footing bond material");
        requireEqual(pack.materials[pack.bonds[4].material].name,
                     std::string("drywall-track"), "facade clip material");

        // Bonds 0 and 1 omit `m` entirely — absent must mean 0, not "unset".
        requireEqual<std::uint32_t>(pack.bonds[0].material, 0u, "omitted m defaults to 0");
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    std::printf("scene pack conformance (C++) passed\n");
    return 0;
}
