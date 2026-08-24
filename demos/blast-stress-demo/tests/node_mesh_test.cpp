// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Coverage for ScenePack::nodeMeshes and its render wiring.
//
// `nodeMeshes` lets a node draw as its real shape (e.g. a Voronoi-fractured
// shard's actual convex hull) instead of its AABB box — see
// SCENE_PACK_FORMAT.md and mini_city_main.cpp's VisualActor construction.
// This test pins two things: the loader correctly parses a pack that carries
// real geometry for every node (fractured-tower.json), and it correctly
// treats an absent/null `nodeMeshes` array as "no mesh anywhere" without
// erroring, since most packs (including the calibrated reference building)
// are pure box geometry and never populate this field.

#include "../scene_pack.h"

#include <cstdio>
#include <fstream>
#include <sstream>
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
        throw std::runtime_error("node mesh test failed: " + message);
    }
}

// A pack that carries real geometry for every node: fractured-tower.json,
// where every node (even mass-0 supports) has a populated nodeMeshes entry.
void testPopulatedNodeMeshesParse(const std::string& fracturedTowerPath)
{
    const ScenePack pack = loadScenePack(fracturedTowerPath);
    require(pack.nodeMeshes.size() == pack.nodes.size(), "nodeMeshes must be padded to node count");

    std::size_t present = 0;
    for (std::size_t i = 0; i < pack.nodeMeshes.size(); ++i)
    {
        const blast_demo::SceneMesh& mesh = pack.nodeMeshes[i];
        if (!mesh.present)
        {
            continue;
        }
        ++present;
        require(!mesh.positions.empty(), "a present mesh must have vertices");
        require(mesh.positions.size() == mesh.normals.size(), "positions/normals must be parallel");
        require(!mesh.indices.empty() && mesh.indices.size() % 3 == 0,
                "indices must be whole triangles");
        for (std::uint32_t index : mesh.indices)
        {
            require(index < mesh.positions.size(), "index must reference a real vertex");
        }
    }
    require(present == pack.nodes.size(), "fractured-tower.json should mesh every node");
}

// A pack with no `nodeMeshes` key at all must load fine, with every entry
// present=false — the common case (pure box geometry) must not require
// authoring this field, and must not misinterpret its absence as an error.
void testAbsentNodeMeshesIsAllBoxes(const std::string& referenceBuildingPath)
{
    const ScenePack pack = loadScenePack(referenceBuildingPath);
    require(pack.nodeMeshes.size() == pack.nodes.size(), "nodeMeshes must be padded to node count");
    for (const blast_demo::SceneMesh& mesh : pack.nodeMeshes)
    {
        require(!mesh.present, "a box-only pack must report no meshed nodes");
    }
}

// A mismatched-length nodeMeshes array must be ignored (not crash, not throw)
// rather than partially applied — the loader is explicit that this is
// defensive: a pack that doesn't care about non-box visuals shouldn't need to
// know this field's exact shape.
void testMismatchedLengthIsIgnored(const std::string& referenceBuildingPath)
{
    std::ifstream input(referenceBuildingPath);
    require(input.good(), "could not open reference building fixture");
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string json = buffer.str();

    const std::string needle = "\"scenario\":";
    const std::size_t at = json.find(needle);
    require(at != std::string::npos, "fixture is missing a scenario key");
    const std::string injected =
        "\"nodeMeshes\": [{\"positions\": [0,0,0], \"normals\": [0,1,0], \"indices\": [0,0,0]}], " + needle;
    json.replace(at, needle.size(), injected);

    const std::string tempPath = referenceBuildingPath + ".mismatched-node-meshes-test.json";
    {
        std::ofstream output(tempPath);
        output << json;
    }
    const ScenePack pack = loadScenePack(tempPath);
    std::remove(tempPath.c_str());

    require(pack.nodeMeshes.size() == pack.nodes.size(), "nodeMeshes must be padded to node count");
    for (const blast_demo::SceneMesh& mesh : pack.nodeMeshes)
    {
        require(!mesh.present, "a length-mismatched nodeMeshes array must be ignored entirely");
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: node_mesh_test <fractured-tower.json> <reference-building.json>\n");
        return 2;
    }
    try
    {
        testPopulatedNodeMeshesParse(argv[1]);
        testAbsentNodeMeshesIsAllBoxes(argv[2]);
        testMismatchedLengthIsIgnored(argv[2]);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("node mesh test passed\n");
    return 0;
}
