#!/usr/bin/env python3
"""
diagnose_overlap.py — quantify collider overlap in a GLB, the root cause of the
"detached chunks explode" problem.

The runtime builds ONE convex hull per piece. While bonded into one rigid body
those hulls are inert (same-body collider pairs don't collide), but the instant a
piece detaches it deeply interpenetrates its neighbours and the contact solver
resolves that as a blast. This script measures, at the part level, how much the
parts' convex hulls overlap — so we know how bad it is and whether convex
decomposition (CoACD) is warranted.

Pipeline mirror: load GLB -> world-space meshes -> split into connected
components (the runtime's splitConnectedComponents) -> drop the ground plane ->
convex hull per part -> pairwise penetration depth via fcl.

    python3 diagnose_overlap.py path/to/model.glb
"""
import sys
import numpy as np
import trimesh
from trimesh import collision


# Mirror the runtime splitConnectedComponents (glb-vehicle.ts): weld at 1e-4, drop
# components below MIN_TRIS triangles, keep at most MAX_COMPONENTS (largest) per mesh.
MIN_TRIS = 6
MAX_COMPONENTS = 24


def load_parts(glb_path):
    scene = trimesh.load(glb_path, process=False, force="scene")
    parts = []  # (name, world-space Trimesh component)
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        mesh.merge_vertices(digits_vertex=4)
        comps = [c for c in mesh.split(only_watertight=False) if len(c.faces) >= MIN_TRIS]
        if len(comps) == 0:
            comps = [mesh]
        comps.sort(key=lambda c: -len(c.faces))
        comps = comps[:MAX_COMPONENTS]
        for i, comp in enumerate(comps):
            parts.append((f"{geom_name}#{i}" if len(comps) > 1 else geom_name, comp))
    return parts


def drop_ground(parts):
    # Drop any part whose footprint is much larger than the median (the ground plane).
    if not parts:
        return parts
    dias = [float(np.linalg.norm(p.extents)) for _, p in parts]
    med = float(np.median(dias))
    kept = [(n, p) for (n, p), d in zip(parts, dias) if d <= max(med * 4.0, 20.0)]
    return kept


def convex(p):
    try:
        return p.convex_hull
    except Exception as e:
        print(f"[diagnose] WARNING convex_hull failed ({type(e).__name__}: {e}); using raw mesh")
        return p


def main():
    glb = sys.argv[1] if len(sys.argv) > 1 else "assets/buggy.glb"
    parts = drop_ground(load_parts(glb))
    hulls = [(n, convex(p)) for n, p in parts]
    print(f"\n[diagnose] {glb}: {len(hulls)} parts (after split + ground drop)")

    mgr = collision.CollisionManager()
    for i, (n, h) in enumerate(hulls):
        mgr.add_object(str(i), h)

    hit, names, data = mgr.in_collision_internal(return_names=True, return_data=True)
    # Aggregate penetration depth per overlapping pair.
    pair_depth = {}
    for c in data:
        a, b = sorted(c.names)
        d = abs(float(getattr(c, "depth", 0.0)))
        pair_depth[(a, b)] = max(pair_depth.get((a, b), 0.0), d)

    npairs = len(hulls) * (len(hulls) - 1) // 2
    overlapping = len(pair_depth)
    depths = sorted(pair_depth.values(), reverse=True)
    print(f"[diagnose] overlapping part-pairs: {overlapping} / {npairs}")
    if depths:
        print(f"[diagnose] penetration depth (m): max={depths[0]:.3f} "
              f"mean={np.mean(depths):.3f} median={np.median(depths):.3f}")
        print("[diagnose] worst 12 overlaps (depth m | part A <-> part B):")
        worst = sorted(pair_depth.items(), key=lambda kv: -kv[1])[:12]
        for (a, b), d in worst:
            print(f"    {d:6.3f}  {hulls[int(a)][0]:>22}  <->  {hulls[int(b)][0]}")
    # Per-part overlap count (which parts are the worst offenders).
    cnt = {}
    for (a, b) in pair_depth:
        cnt[a] = cnt.get(a, 0) + 1
        cnt[b] = cnt.get(b, 0) + 1
    if cnt:
        print("[diagnose] most-overlapping parts (overlap count | part | size m):")
        for k, c in sorted(cnt.items(), key=lambda kv: -kv[1])[:10]:
            ext = hulls[int(k)][1].extents
            print(f"    {c:3d}  {hulls[int(k)][0]:>22}  ({ext[0]:.2f},{ext[1]:.2f},{ext[2]:.2f})")


if __name__ == "__main__":
    main()
