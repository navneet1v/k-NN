"""
Shuffle an ANN HDF5 dataset's train order (decorrelate id from vector space) WITHOUT recomputing
ground truth — only id relabeling.

Why: some benchmark datasets (e.g. documents-1m.hdf5) are pre-sorted so that near-neighbors have
adjacent ids. Their natural-order baseline is already near-optimal and hides the locality-reordering
benefit. This tool bakes a random permutation into a new file so the on-disk order is uncorrelated with
vector space — the realistic random-arrival case — and you can then ingest it normally (no --shuffle
flag) and benchmark honestly.

What it does (bijective relabel, ground truth preserved exactly):
  perm = random permutation of [0, N)             (fixed --seed for reproducibility)
  new_train[i]      = train[perm[i]]              # output row i holds the vector formerly at perm[i]
  inv_perm[perm[i]] = i                           # a vector formerly at id t now lives at id inv_perm[t]
  new_neighbors     = inv_perm[neighbors]         # remap GT ids to the new layout (NO recompute)
  distances, test   = copied unchanged            # same vectors/answers, just relabeled ids

Because we only relabel ids, every query's true neighbor SET and their distances are identical — the
GT is exact, not approximated.

Usage:
    pip install h5py numpy
    python scripts/shuffle_dataset.py --input scripts/documents-1m.hdf5 --output scripts/documents-1m-shuffled.hdf5
    # then verify it's now realistic:
    python scripts/dataset_order_report.py --dataset scripts/documents-1m-shuffled.hdf5
"""

import argparse

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Shuffle train order of an ANN HDF5 dataset and remap GT ids (no recompute)")
    p.add_argument("--input", required=True, help="source HDF5 (train, neighbors, test, distances, ...)")
    p.add_argument("--output", required=True, help="destination HDF5 to create")
    p.add_argument("--seed", type=int, default=42, help="permutation seed (reproducible)")
    p.add_argument("--chunk", type=int, default=10000, help="rows per chunk when gathering train (memory bound)")
    p.add_argument("--train-key", default="train")
    p.add_argument("--neighbors-key", default="neighbors")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        keys = list(fin.keys())
        train = fin[args.train_key]
        n, dim = train.shape[0], train.shape[1]
        print(f"input: {args.input}  keys={keys}  train={train.shape} dtype={train.dtype}")

        # Permutation and its inverse.
        perm = rng.permutation(n)                       # new_train[i] = train[perm[i]]
        inv_perm = np.empty(n, dtype=np.int64)
        inv_perm[perm] = np.arange(n, dtype=np.int64)   # old id -> new id
        print(f"permutation seed={args.seed}  N={n}")

        # 1) Shuffled train, written in chunks (h5py fancy-read needs increasing indices, so sort/unsort).
        out_train = fout.create_dataset(args.train_key, shape=train.shape, dtype=train.dtype,
                                        chunks=train.chunks, compression=train.compression)
        for s in range(0, n, args.chunk):
            e = min(s + args.chunk, n)
            idx = perm[s:e]                             # source ids for output rows [s,e)
            order = np.argsort(idx)                     # h5py needs increasing, unique (perm => unique)
            block = train[np.sort(idx)]                 # gather sorted
            restore = np.empty_like(order)
            restore[order] = np.arange(len(order))
            out_train[s:e] = block[restore]             # put back into output order
            if e % (args.chunk * 20) == 0 or e == n:
                print(f"  train {e}/{n}")

        # 2) Remap neighbor (GT) ids: a neighbor formerly at id t now lives at inv_perm[t]. No recompute.
        if args.neighbors_key in fin:
            gt = np.asarray(fin[args.neighbors_key])
            gt_dtype = fin[args.neighbors_key].dtype
            remapped = inv_perm[gt.astype(np.int64)].astype(gt_dtype)
            fout.create_dataset(args.neighbors_key, data=remapped, compression=fin[args.neighbors_key].compression)
            print(f"  remapped {args.neighbors_key} {gt.shape} (GT preserved, ids relabeled)")
        else:
            print(f"  WARNING: no '{args.neighbors_key}' dataset; nothing to remap")

        # 3) Everything else (test, distances, ...) copied unchanged — it does not reference train ids.
        for k in keys:
            if k in (args.train_key, args.neighbors_key):
                continue
            fin.copy(k, fout)
            print(f"  copied '{k}' unchanged")

        # 4) Spot-check: output row inv_perm[t] must equal the original vector at id t, for a few t.
        rc = rng.choice(n, size=min(5, n), replace=False)
        ok = True
        for t in rc:
            if not np.array_equal(np.asarray(out_train[int(inv_perm[t])]), np.asarray(train[int(t)])):
                ok = False
                break
        print(f"  verify vector-preservation on {len(rc)} samples: {'OK' if ok else 'FAILED'}")

    print(f"\nwrote {args.output}")
    print(f"Next: python scripts/dataset_order_report.py --dataset {args.output}   # expect REALISTIC")
    print("Then ingest it normally (no --shuffle-ingest needed) — the on-disk order is already decorrelated.")


if __name__ == "__main__":
    main()
