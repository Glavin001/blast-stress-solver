// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "../scene_pack.h"

#include <cmath>
#include <cstdio>
#include <exception>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: scene_pack_test <scene-pack.json>\n");
        return 2;
    }
    try
    {
        const blast_demo::ScenePack pack = blast_demo::loadScenePack(argv[1]);
        if (pack.nodes.size() != 204 || pack.bonds.size() != 546)
        {
            std::fprintf(
                stderr,
                "unexpected fractured-tower topology: %zu nodes, %zu bonds\n",
                pack.nodes.size(),
                pack.bonds.size());
            return 1;
        }
        std::size_t supports = 0;
        std::size_t convexes = 0;
        std::size_t simplified = 0;
        for (const blast_demo::SceneNode& node : pack.nodes)
        {
            supports += node.mass == 0.0f ? 1 : 0;
            convexes += node.collider.kind == blast_demo::SceneColliderKind::ConvexHull ? 1 : 0;
            simplified += node.collider.sourcePointCount > node.collider.points.size() ? 1 : 0;
            if (!node.centroid.isFinite() || !std::isfinite(node.mass) || node.volume <= 0.0f)
            {
                std::fprintf(stderr, "scene contains a non-finite or non-positive node\n");
                return 1;
            }
            if (node.collider.points.size() > 64)
            {
                std::fprintf(stderr, "scene contains a convex hull above the GPU 64-point limit\n");
                return 1;
            }
        }
        if (supports == 0 || convexes != 168 || simplified == 0)
        {
            std::fprintf(
                stderr,
                "unexpected support/convex counts: %zu supports, %zu convexes, %zu simplified\n",
                supports,
                convexes,
                simplified);
            return 1;
        }
        std::printf(
            "ScenePack OK: %zu nodes, %zu bonds, %zu supports, %zu convex hulls, %zu simplified\n",
            pack.nodes.size(),
            pack.bonds.size(),
            supports,
            convexes,
            simplified);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "ScenePack test failed: %s\n", error.what());
        return 1;
    }
}
