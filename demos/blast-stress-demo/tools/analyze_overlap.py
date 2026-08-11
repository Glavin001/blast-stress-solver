#!/usr/bin/env python3
"""Measure real solid-solid interpenetration in a ScenePack.

Bodies that interpenetrate while bonded are loaded springs: the bond hides the
penetration, and the contact solver ejects both bodies the moment it breaks.
That reads as an explosion rather than a collapse, so a pack meant to come
apart under load must be authored flush, not embedded.

Bounding boxes are only a broadphase filter here. Slanted convex shards have
overlapping AABBs even when the hulls are disjoint, so every candidate pair is
resolved by Monte-Carlo sampling against the true hull planes (for nodes with
a nodeMeshes entry) or the true box (for everything else). Points are tested
strictly inside both shapes, so flush face-to-face contact counts as zero.
"""
import argparse
import itertools
import json
import random
import sys
from collections import defaultdict

SAMPLES = 3000
GRID_CELL = 2.0
SKIN = 1e-4  # shapes must overlap by more than this to count


def hull_planes(node, mesh):
    centre = node["centroid"]
    pos, idx = mesh["positions"], mesh["indices"]
    verts = [(pos[k * 3] + centre["x"], pos[k * 3 + 1] + centre["y"], pos[k * 3 + 2] + centre["z"])
             for k in range(len(pos) // 3)]
    mx = sum(v[0] for v in verts) / len(verts)
    my = sum(v[1] for v in verts) / len(verts)
    mz = sum(v[2] for v in verts) / len(verts)
    planes = []
    for t in range(0, len(idx), 3):
        a, b, c = verts[idx[t]], verts[idx[t + 1]], verts[idx[t + 2]]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        w = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
        if length < 1e-12:
            continue
        n = (n[0] / length, n[1] / length, n[2] / length)
        if n[0] * (mx - a[0]) + n[1] * (my - a[1]) + n[2] * (mz - a[2]) > 0:
            n = (-n[0], -n[1], -n[2])
        planes.append((n, n[0] * a[0] + n[1] * a[1] + n[2] * a[2]))
    return planes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack")
    parser.add_argument("--max-pairs", type=int, default=100,
                        help="pairs sampled per role combination (result is scaled up)")
    parser.add_argument("--require-clean", action="store_true",
                        help="exit non-zero if any true interpenetration is found")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="m^3 of total interpenetration tolerated by --require-clean")
    args = parser.parse_args()
    random.seed(11)

    pack = json.load(open(args.pack))
    scenario = pack["scenario"]
    nodes, types, sizes = scenario["nodes"], scenario["nodeTypes"], scenario["nodeSizes"]
    meshes = pack.get("nodeMeshes") or scenario.get("nodeMeshes") or [None] * len(nodes)

    def box(i):
        c, s = nodes[i]["centroid"], sizes[i]
        return (c["x"] - s["x"] / 2, c["y"] - s["y"] / 2, c["z"] - s["z"] / 2,
                c["x"] + s["x"] / 2, c["y"] + s["y"] / 2, c["z"] + s["z"] / 2)

    boxes = [box(i) for i in range(len(nodes))]

    def overlap_box(a, b):
        r = [max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]),
             min(a[3], b[3]), min(a[4], b[4]), min(a[5], b[5])]
        wide = r[3] > r[0] + SKIN and r[4] > r[1] + SKIN and r[5] > r[2] + SKIN
        return r if wide else None

    grid = defaultdict(list)
    for i, b in enumerate(boxes):
        for gx in range(int(b[0] // GRID_CELL), int(b[3] // GRID_CELL) + 1):
            for gy in range(int(b[1] // GRID_CELL), int(b[4] // GRID_CELL) + 1):
                for gz in range(int(b[2] // GRID_CELL), int(b[5] // GRID_CELL) + 1):
                    grid[(gx, gy, gz)].append(i)

    candidates = defaultdict(set)
    for bucket in grid.values():
        for i, j in itertools.combinations(sorted(set(bucket)), 2):
            if overlap_box(boxes[i], boxes[j]):
                candidates[tuple(sorted((types[i], types[j])))].add((i, j))

    cache = {}

    def tester(i):
        if meshes[i]:
            if i not in cache:
                cache[i] = hull_planes(nodes[i], meshes[i])
            planes = cache[i]
            return lambda q: all(n[0] * q[0] + n[1] * q[1] + n[2] * q[2] <= d - SKIN
                                 for n, d in planes)
        b = boxes[i]
        return lambda q: (b[0] + SKIN <= q[0] <= b[3] - SKIN
                          and b[1] + SKIN <= q[1] <= b[4] - SKIN
                          and b[2] + SKIN <= q[2] <= b[5] - SKIN)

    print(f"{'pair':26s} {'candidates':>11} {'sampled':>8} {'overlapping':>12} {'est m^3':>10}")
    grand = 0.0
    for key in sorted(candidates, key=lambda k: -len(candidates[k])):
        chosen = sorted(candidates[key])
        random.shuffle(chosen)
        chosen = chosen[:args.max_pairs]
        total, hits = 0.0, 0
        for i, j in chosen:
            r = overlap_box(boxes[i], boxes[j])
            volume = (r[3] - r[0]) * (r[4] - r[1]) * (r[5] - r[2])
            ti, tj = tester(i), tester(j)
            inside = 0
            for _ in range(SAMPLES):
                q = (random.uniform(r[0], r[3]), random.uniform(r[1], r[4]), random.uniform(r[2], r[5]))
                if ti(q) and tj(q):
                    inside += 1
            got = volume * inside / SAMPLES
            if got > 1e-5:
                hits += 1
            total += got
        estimate = total * len(candidates[key]) / len(chosen)
        grand += estimate
        print(f"{str(key):26s} {len(candidates[key]):11d} {len(chosen):8d} {hits:12d} {estimate:10.3f}")

    print(f"\ntotal true interpenetration: {grand:.3f} m^3")
    if args.require_clean and grand > args.tolerance:
        print(f"FAIL: exceeds tolerance {args.tolerance} m^3")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
