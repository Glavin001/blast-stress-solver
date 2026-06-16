#!/usr/bin/env python3
"""
decompose_coacd.py — prove convex decomposition (CoACD) removes the collider
overlap that causes detached-chunk explosions.

For each part we compare two collider strategies:
  BEFORE: one convex hull per part (what the runtime does today) — bulges into
          neighbours for any concave part.
  AFTER:  CoACD convex decomposition — a set of tight convex hulls that hug the
          shape, so they don't reach into neighbouring parts.

It reports cross-part penetration (same-part hulls are excluded: pieces of one
part live on one rigid body and never collide with each other) for both, so we
can see the overlap collapse.

    python3 decompose_coacd.py assets/buggy.glb [--threshold 0.05] [--limit N]
"""
import sys
import time
import numpy as np
import trimesh
from trimesh import collision
import coacd

from diagnose_overlap import load_parts, drop_ground, convex


def coacd_hulls(mesh, threshold):
    cm = coacd.Mesh(mesh.vertices, mesh.faces)
    pieces = coacd.run_coacd(cm, threshold=threshold)
    hulls = []
    for verts, faces in pieces:
        h = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            h = h.convex_hull
        except Exception:
            pass
        hulls.append(h)
    return hulls


def cross_part_overlap(part_hulls):
    """part_hulls: list of (part_id, hull). Returns (n_overlap_pairs, max_depth, mean_depth)."""
    mgr = collision.CollisionManager()
    owner = {}
    for i, (pid, h) in enumerate(part_hulls):
        mgr.add_object(str(i), h)
        owner[str(i)] = pid
    _, data = mgr.in_collision_internal(return_names=False, return_data=True)
    pair_depth = {}
    for c in data:
        a, b = c.names
        if owner[a] == owner[b]:
            continue  # same part -> same body -> never collides
        key = tuple(sorted((a, b)))
        pair_depth[key] = max(pair_depth.get(key, 0.0), abs(float(getattr(c, "depth", 0.0))))
    if not pair_depth:
        return 0, 0.0, 0.0
    d = list(pair_depth.values())
    return len(pair_depth), max(d), float(np.mean(d))


def main():
    args = sys.argv[1:]
    glb = args[0] if args and not args[0].startswith("--") else "assets/buggy.glb"
    threshold = float(args[args.index("--threshold") + 1]) if "--threshold" in args else 0.05
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 0

    parts = drop_ground(load_parts(glb))
    if limit:
        # keep the largest parts (most likely the concave offenders)
        parts.sort(key=lambda np_: -float(np.linalg.norm(np_[1].extents)))
        parts = parts[:limit]
    print(f"\n[coacd] {glb}: decomposing {len(parts)} parts (threshold={threshold})")

    before = [(i, convex(p)) for i, (_, p) in enumerate(parts)]
    coacd.set_log_level("error")
    after = []
    t0 = time.time()
    total_hulls = 0
    for i, (name, p) in enumerate(parts):
        try:
            hs = coacd_hulls(p, threshold)
        except Exception as e:
            hs = [convex(p)]
        total_hulls += len(hs)
        for h in hs:
            after.append((i, h))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(parts)} parts, {total_hulls} hulls, {time.time()-t0:.0f}s")

    nb, mb, ab = cross_part_overlap(before)
    na, ma, aa = cross_part_overlap(after)
    print(f"\n[coacd] hulls: {len(parts)} (1/part)  ->  {total_hulls} (CoACD, {total_hulls/len(parts):.1f}/part)")
    print(f"[coacd] BEFORE (one hull/part):  overlapping pairs={nb:5d}  maxDepth={mb:.3f}m  meanDepth={ab:.3f}m")
    print(f"[coacd] AFTER  (CoACD hulls):    overlapping pairs={na:5d}  maxDepth={ma:.3f}m  meanDepth={aa:.3f}m")
    if nb:
        print(f"[coacd] overlap pairs reduced {100*(1-na/max(nb,1)):.0f}%, "
              f"max depth {100*(1-ma/max(mb,1e-9)):.0f}%")


if __name__ == "__main__":
    main()
