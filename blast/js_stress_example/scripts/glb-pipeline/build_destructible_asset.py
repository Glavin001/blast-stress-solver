#!/usr/bin/env python3
"""
build_destructible_asset.py — turn a GLB into NON-OVERLAPPING convex collider
pieces, the heart of the high-quality-destructible pipeline.

Steps:
  1. Split the GLB into parts (mirrors the runtime splitConnectedComponents).
  2. De-interpenetrate: where two parts overlap (the artist modelled wheels inside
     wheel-wells, frame inside body, ...), the SMALLER part yields — we boolean-
     subtract the larger part from it, removing the hidden interior so the
     colliders don't share space. Guarded: a trim that destroys a part is skipped.
  3. CoACD-decompose each (trimmed) part into tight convex hulls that hug the shape
     instead of one bulging hull.
  4. Report residual cross-part overlap (should approach zero) and write an asset
     JSON the runtime loads as render+collider geometry.

    python3 build_destructible_asset.py assets/buggy.glb -o assets/buggy.pieces.json \
        [--threshold 0.04] [--no-deint] [--limit N]
"""
import sys
import json
import time
import numpy as np
import trimesh
from trimesh import collision
import coacd

from diagnose_overlap import load_parts, drop_ground, convex


def aabb_overlap(a, b, pad=0.0):
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    return np.all(amax + pad >= bmin) and np.all(bmax + pad >= amin)


def deinterpenetrate(parts):
    """parts: list of (name, mesh). Bigger parts keep their volume; a smaller part
    yields the region it shares with any bigger part. Returns trimmed meshes."""
    vols = [max(float(p.volume), 1e-9) for _, p in parts]
    order = sorted(range(len(parts)), key=lambda i: -vols[i])  # big -> small
    rank = {idx: r for r, idx in enumerate(order)}
    out = []
    trims = 0
    for i, (name, m) in enumerate(parts):
        trimmed = m
        v0 = vols[i]
        for j, (_, mj) in enumerate(parts):
            if i == j or rank[j] >= rank[i]:
                continue  # only subtract STRICTLY larger parts
            if not aabb_overlap(trimmed, mj):
                continue
            try:
                diff = trimmed.difference(mj)
            except Exception:
                diff = None
            if diff is None or diff.is_empty:
                continue
            vd = float(diff.volume)
            # Guard: accept the trim only if it removed something but didn't gut the
            # part (boolean on open artist meshes can erase it).
            if 0.02 * v0 < vd < 0.999 * v0:
                trimmed = diff
                trims += 1
        out.append((name, trimmed))
    print(f"[build] de-interpenetration: {trims} trims applied")
    return out


def coacd_pieces(mesh, threshold):
    cm = coacd.Mesh(mesh.vertices, mesh.faces)
    pieces = coacd.run_coacd(cm, threshold=threshold)
    hulls = []
    for verts, faces in pieces:
        h = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
        try:
            h = h.convex_hull
        except Exception:
            pass
        if len(h.vertices) >= 4:
            hulls.append(h)
    return hulls or [convex(mesh)]


def cross_part_overlap(part_hulls):
    mgr = collision.CollisionManager()
    owner = {}
    for i, (pid, h) in enumerate(part_hulls):
        mgr.add_object(str(i), h)
        owner[str(i)] = pid
    _, data = mgr.in_collision_internal(return_names=False, return_data=True)
    pd = {}
    for c in data:
        a, b = c.names
        if owner[a] == owner[b]:
            continue
        pd[tuple(sorted((a, b)))] = max(pd.get(tuple(sorted((a, b))), 0.0), abs(float(getattr(c, "depth", 0.0))))
    if not pd:
        return 0, 0.0, 0.0
    d = list(pd.values())
    return len(pd), max(d), float(np.mean(d))


def main():
    args = sys.argv[1:]
    glb = next((a for a in args if not a.startswith("--") and a.endswith(".glb")), "assets/buggy.glb")
    out = args[args.index("-o") + 1] if "-o" in args else None
    threshold = float(args[args.index("--threshold") + 1]) if "--threshold" in args else 0.04
    do_deint = "--no-deint" not in args
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 0

    parts = drop_ground(load_parts(glb))
    if limit:
        parts.sort(key=lambda nm: -float(np.linalg.norm(nm[1].extents)))
        parts = parts[:limit]
    print(f"\n[build] {glb}: {len(parts)} parts  (deint={do_deint}, threshold={threshold})")

    if do_deint:
        parts = deinterpenetrate(parts)

    coacd.set_log_level("error")
    t0 = time.time()
    asset_parts = []
    part_hulls = []  # (part_index, hull) for overlap measurement
    for i, (name, m) in enumerate(parts):
        hulls = coacd_pieces(m, threshold)
        pieces = [{"vertices": np.asarray(h.vertices, np.float32).round(5).tolist(),
                   "faces": np.asarray(h.faces, np.int32).tolist()} for h in hulls]
        c = m.bounds.mean(axis=0)
        asset_parts.append({"name": name, "centroid": [float(x) for x in c],
                            "extents": [float(x) for x in m.extents], "pieces": pieces})
        for h in hulls:
            part_hulls.append((i, h))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(parts)} parts, {len(part_hulls)} hulls, {time.time()-t0:.0f}s")

    n, mx, mn = cross_part_overlap(part_hulls)
    print(f"\n[build] {len(asset_parts)} parts -> {len(part_hulls)} hulls in {time.time()-t0:.0f}s")
    print(f"[build] residual cross-part overlap: pairs={n}  maxDepth={mx:.3f}m  meanDepth={mn:.3f}m")

    if out:
        with open(out, "w") as f:
            json.dump({"source": glb, "threshold": threshold, "deinterpenetrated": do_deint,
                       "parts": asset_parts}, f)
        print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()
