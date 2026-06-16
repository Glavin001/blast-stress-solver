#!/usr/bin/env python3
"""
_coacd_worker.py — run ONE CoACD decomposition in an isolated process.

CoACD is a C++ library that can hard-abort (std::terminate, "unexpected code path
was hit") on degenerate input — an in-process try/except cannot catch that, so it
would kill the whole build. Running it here, one mesh per process, turns a crash
into a non-zero exit the parent can detect, report, and recover from.

    python3 _coacd_worker.py in.npz out.npz <threshold>
in.npz:  arrays v (Nx3 float), f (Mx3 int)
out.npz: n (int), then v0,f0,v1,f1,... per convex piece
"""
import sys
import numpy as np
import coacd


def main():
    inp, outp, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    d = np.load(inp)
    coacd.set_log_level("error")
    pieces = coacd.run_coacd(coacd.Mesh(d["v"], d["f"]), threshold=thr,
                             preprocess_resolution=30, resolution=1000, mcts_iterations=60,
                             mcts_max_depth=2, max_convex_hull=24, decimate=True, max_ch_vertex=32)
    arrs = {"n": np.int32(len(pieces))}
    for i, (v, f) in enumerate(pieces):
        arrs[f"v{i}"] = np.asarray(v, np.float64)
        arrs[f"f{i}"] = np.asarray(f, np.int32)
    np.savez(outp, **arrs)


if __name__ == "__main__":
    main()
