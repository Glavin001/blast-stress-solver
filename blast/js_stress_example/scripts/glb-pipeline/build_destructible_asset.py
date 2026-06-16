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
import os
import json
import time
import pickle
import tempfile
import subprocess
import numpy as np
import trimesh
from trimesh import collision
from scipy.spatial import ConvexHull, HalfspaceIntersection
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
    bool_errors = {}
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
            except Exception as e:
                bool_errors[type(e).__name__] = bool_errors.get(type(e).__name__, 0) + 1
                continue
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
    if bool_errors:
        print(f"[build] WARNING de-interpenetration: boolean errors {bool_errors} "
              "(open/non-watertight parts cannot be subtracted)")
    return out


def inset_hull(hull, delta):
    """Shrink a convex hull by moving every face plane inward by `delta` (a true
    inset, uniform in distance regardless of piece size). Removes residual convex
    overshoot so neighbouring pieces no longer share space. Returns the original if
    the piece is too thin to inset."""
    try:
        ch = ConvexHull(hull.vertices)
    except Exception as e:
        print(f"[build] WARNING inset_hull: ConvexHull failed ({type(e).__name__}: {e}); kept whole")
        return hull
    hs = ch.equations.copy()        # A x + b <= 0 (A outward-normal, |A|=1)
    hs[:, -1] += delta              # push each plane inward by delta
    interior = hull.vertices.mean(axis=0)
    # The centroid must stay strictly inside the inset region, else the piece is
    # thinner than 2*delta in some direction — an expected case: keep it whole.
    if np.any(hs[:, :3] @ interior + hs[:, -1] >= -1e-6):
        return hull
    try:
        pts = HalfspaceIntersection(hs, interior).intersections
        out = trimesh.Trimesh(vertices=pts).convex_hull
    except Exception as e:
        print(f"[build] WARNING inset_hull: halfspace inset failed ({type(e).__name__}: {e}); kept whole")
        return hull
    return out if len(out.vertices) >= 4 else hull


def clip_deinterpenetrate(pieces, iters=15, margin=0.01):
    """pieces: list of [part_index, priority, hull]. Where two pieces from DIFFERENT
    parts overlap, clip the lower-priority one along the contact plane (a convex hull
    sliced by a plane stays convex), removing its penetrating lobe + a `margin` so it
    ends with a gap rather than merely touching. Unlike inset this works on thin shell
    pieces. Iterates until clean or `iters` reached.

    Failures are surfaced, never swallowed: slice errors are counted by exception
    type and reported, and any error that wipes out *every* attempt (e.g. a missing
    backend) raises instead of silently doing nothing."""
    hulls = [p[2] for p in pieces]
    slice_attempts = 0
    slice_errors = {}      # exception type name -> count
    slice_too_small = 0    # clip would erase the piece (kept whole)
    for _ in range(iters):
        mgr = collision.CollisionManager()
        for i, h in enumerate(hulls):
            mgr.add_object(str(i), h)
        _, data = mgr.in_collision_internal(return_names=False, return_data=True)
        worst = {}  # deepest contact per overlapping cross-part pair
        for c in data:
            a, b = (int(x) for x in c.names)  # c.names is a set
            if pieces[a][0] == pieces[b][0]:
                continue
            key = (a, b)
            d = abs(float(getattr(c, "depth", 0.0)))
            if d > worst.get(key, (0.0,))[0]:
                worst[key] = (d, np.asarray(c.point, float), np.asarray(c.normal, float))
        if not worst:
            break
        changed = False
        for (a, b), (d, point, normal) in worst.items():
            lo = a if pieces[a][1] <= pieces[b][1] else b
            other = b if lo == a else a
            to_other = np.asarray(hulls[other].centroid) - np.asarray(hulls[lo].centroid)
            n = normal if np.dot(normal, to_other) > 0 else -normal
            origin = point - margin * n  # clip slightly past contact -> gap, not touch
            slice_attempts += 1
            try:
                s = hulls[lo].slice_plane(plane_origin=origin, plane_normal=-n, cap=True)
            except Exception as e:
                slice_errors[type(e).__name__] = slice_errors.get(type(e).__name__, 0) + 1
                continue
            if s is None or s.is_empty or len(s.vertices) < 4 or s.volume <= 1e-7:
                slice_too_small += 1
                continue
            hulls[lo] = s.convex_hull
            changed = True
        if not changed:
            break
    if slice_errors:
        msg = ", ".join(f"{k}x{v}" for k, v in slice_errors.items())
        # If EVERY slice errored, the step is silently a no-op — fail loudly.
        if sum(slice_errors.values()) >= slice_attempts and slice_attempts > 0:
            raise RuntimeError(f"clip_deinterpenetrate: all {slice_attempts} slices failed ({msg}). "
                               "A backend is likely missing (e.g. shapely for slice_plane).")
        print(f"[build] WARNING clip: {sum(slice_errors.values())}/{slice_attempts} slices errored ({msg})")
    if slice_too_small:
        print(f"[build] clip: {slice_too_small} clips skipped (would erase a too-thin piece)")
    for i, h in enumerate(hulls):
        pieces[i][2] = h
    return pieces


def boolean_deinterpenetrate(flat, threshold, dilate=0.004):
    """flat: [part_index, priority, hull] where hulls are the WATERTIGHT CONVEX CoACD
    pieces — which is the whole point: boolean ops failed on the open-shell source
    parts, but succeed on these solid convex hulls.

    Process pieces smallest-volume first (small detail parts are authoritative and
    keep their full shape); each piece has every already-placed SMALLER cross-part
    piece subtracted from it, so the larger piece gets the exact overlap notched out.
    The non-convex result is re-CoACD'd back into convex pieces. Exact de-interpen-
    etration, no planar approximation, no small piece losing volume."""
    order = sorted(range(len(flat)), key=lambda i: flat[i][2].volume)
    placed = []   # finalized [part_index, hull] (smaller pieces, authoritative)
    out = []
    bool_err = {}
    n_cut = n_dropped = n_recoacd = 0
    for idx in order:
        pidx, pri, h = flat[idx]
        cut = h
        cut_changed = False
        for (fp, fh) in placed:
            if fp == pidx or not aabb_overlap(cut, fh):
                continue
            try:
                diff = cut.difference(fh)
            except Exception as e:
                bool_err[type(e).__name__] = bool_err.get(type(e).__name__, 0) + 1
                continue
            if diff is None or diff.is_empty:
                cut = None
                break
            if diff.volume < cut.volume - 1e-9:  # actually removed material
                cut = diff
                cut_changed = True
        if cut is None or cut.is_empty or float(cut.volume) < 1e-7:
            n_dropped += 1
            continue
        if cut_changed:
            n_cut += 1
            sub = coacd_pieces_isolated(cut, threshold)  # isolated: notched meshes can crash CoACD
            n_recoacd += len(sub)
            for p in sub:
                placed.append((pidx, p))
                out.append([pidx, pri, p])
        else:
            placed.append((pidx, h))
            out.append([pidx, pri, h])
    print(f"[build] boolean de-interp: {n_cut} pieces notched -> {n_recoacd} re-CoACD pieces, "
          f"{n_dropped} pieces fully consumed")
    if bool_err:
        print(f"[build] WARNING boolean de-interp: errors {bool_err}")
    return out


_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_coacd_worker.py")


def coacd_pieces_isolated(mesh, threshold, timeout=120):
    """CoACD in a subprocess so a C++ hard-abort on degenerate input becomes a
    recoverable, REPORTED error instead of killing the build. Falls back to the
    part's convex hull (loudly) if CoACD crashes/times out."""
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(inp, v=np.asarray(mesh.vertices, np.float64), f=np.asarray(mesh.faces, np.int32))
        try:
            r = subprocess.run([sys.executable, _WORKER, inp, outp, str(threshold)],
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[build] WARNING CoACD timed out (>{timeout}s); using convex hull")
            return [convex(mesh)]
        if r.returncode != 0 or not os.path.exists(outp):
            err = (r.stderr.decode(errors="replace").strip() or "no stderr")[-200:]
            print(f"[build] WARNING CoACD subprocess crashed (rc={r.returncode}): {err}; using convex hull")
            return [convex(mesh)]
        d = np.load(outp)
        hulls = []
        for i in range(int(d["n"])):
            h = trimesh.Trimesh(vertices=d[f"v{i}"], faces=d[f"f{i}"], process=False)
            try:
                h = h.convex_hull
            except Exception as e:
                print(f"[build] WARNING coacd piece convex_hull failed ({type(e).__name__}: {e})")
            if len(h.vertices) >= 4:
                hulls.append(h)
        if not hulls:
            print("[build] WARNING CoACD produced no usable hulls; using the part convex hull")
            return [convex(mesh)]
        return hulls


def coacd_pieces(mesh, threshold):
    cm = coacd.Mesh(mesh.vertices, mesh.faces)
    pieces = coacd.run_coacd(cm, threshold=threshold)
    hulls = []
    for verts, faces in pieces:
        h = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
        try:
            h = h.convex_hull
        except Exception as e:
            print(f"[build] WARNING coacd_pieces: convex_hull failed ({type(e).__name__}: {e}); using raw piece")
        if len(h.vertices) >= 4:
            hulls.append(h)
    if not hulls:
        print(f"[build] WARNING coacd_pieces: CoACD produced no usable hulls; using the part convex hull")
        return [convex(mesh)]
    return hulls


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
    inset = float(args[args.index("--inset") + 1]) if "--inset" in args else 0.02
    do_deint = "--deint" in args  # source-mesh boolean trim (off: open shells don't solid-overlap)
    method = args[args.index("--method") + 1] if "--method" in args else "boolean"
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 0

    parts = drop_ground(load_parts(glb))
    if limit:
        parts.sort(key=lambda nm: -float(np.linalg.norm(nm[1].extents)))
        parts = parts[:limit]
    print(f"\n[build] {glb}: {len(parts)} parts  (threshold={threshold}, inset={inset}, deint={do_deint})")

    if do_deint:
        parts = deinterpenetrate(parts)

    coacd.set_log_level("error")
    t0 = time.time()
    # CoACD every part into tight hulls (cached — it's the slow step), collecting a
    # flat piece list tagged with part index and priority (bigger part = higher
    # priority = keeps its volume).
    cache = f"/tmp/coacd_{os.path.basename(glb)}_{threshold}_{inset}_{limit}.pkl"
    if os.path.exists(cache):
        flat_raw, part_meta = pickle.load(open(cache, "rb"))
        flat = [[pi, pr, trimesh.Trimesh(vertices=v, faces=f, process=False)] for pi, pr, v, f in flat_raw]
        print(f"[build] loaded {len(flat)} CoACD hulls from cache")
    else:
        flat = []  # [part_index, priority, hull]
        part_meta = []
        for i, (name, m) in enumerate(parts):
            hulls = coacd_pieces(m, threshold)
            if inset > 0:
                hulls = [inset_hull(h, inset) for h in hulls]
            pri = float(np.prod(m.extents))
            for h in hulls:
                flat.append([i, pri, h])
            part_meta.append({"name": name, "centroid": [float(x) for x in m.bounds.mean(axis=0)],
                              "extents": [float(x) for x in m.extents]})
            if (i + 1) % 25 == 0:
                print(f"  ...{i+1}/{len(parts)} parts, {len(flat)} hulls, {time.time()-t0:.0f}s")
        flat_raw = [[pi, pr, np.asarray(h.vertices), np.asarray(h.faces)] for pi, pr, h in flat]
        pickle.dump((flat_raw, part_meta), open(cache, "wb"))

    nb, mxb, mnb = cross_part_overlap([(p[0], p[2]) for p in flat])
    print(f"[build] CoACD: {len(parts)} parts -> {len(flat)} hulls in {time.time()-t0:.0f}s")
    print(f"[build] overlap BEFORE {method}: pairs={nb}  maxDepth={mxb:.3f}m  meanDepth={mnb:.3f}m")

    t1 = time.time()
    if method == "boolean":
        flat = boolean_deinterpenetrate(flat, threshold)
    elif method == "clip":
        flat = clip_deinterpenetrate(flat)
    else:
        raise SystemExit(f"unknown --method {method!r} (expected boolean|clip)")
    na, mxa, mna = cross_part_overlap([(p[0], p[2]) for p in flat])
    print(f"[build] overlap AFTER  {method}: pairs={na}  maxDepth={mxa:.3f}m  meanDepth={mna:.3f}m  ({time.time()-t1:.0f}s, {len(flat)} hulls)")

    # Regroup hulls by part into the asset.
    by_part = {}
    for pidx, _pri, h in flat:
        by_part.setdefault(pidx, []).append(h)
    asset_parts = []
    for i, meta in enumerate(part_meta):
        hulls = by_part.get(i, [])
        meta["pieces"] = [{"vertices": np.asarray(h.vertices, np.float32).round(5).tolist(),
                           "faces": np.asarray(h.faces, np.int32).tolist()} for h in hulls]
        asset_parts.append(meta)

    if out:
        with open(out, "w") as f:
            json.dump({"source": glb, "threshold": threshold, "inset": inset,
                       "parts": asset_parts}, f)
        print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()
