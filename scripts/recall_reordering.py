"""
Recall comparison: locality-reordered vs baseline Faiss SQ 1-bit (32x) on OpenSearch.

Ingests the SAME vectors into two 32x on-disk k-NN indices — one with
`index.knn.advanced.reordering_enabled: true` (uses the reordered format) and one
with it false (baseline) — then runs the dataset's test queries against both and
reports recall@k vs the dataset's ground-truth neighbors.

Both indices are identical except for the one setting, so any recall difference is
attributable to the reordered layout + hub-seeded entry point.

Dataset: an HDF5 file with `train` (base vectors), `test` (queries), and `neighbors`
(ground-truth ids into `train`), e.g. scripts/documents-1m.hdf5.

Prereqs:
    pip install requests h5py numpy
    A running OpenSearch (with this k-NN plugin build) reachable at --host.

Example:
    python scripts/recall_reordering.py \
        --host http://localhost:9200 \
        --dataset scripts/documents-1m.hdf5 \
        --num-docs 100000 --num-queries 1000 --k 10 --space-type innerproduct
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import h5py
import numpy as np
import requests


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(path, num_docs, num_queries, search_only):
    """Returns (train, num_train, test, gt). In search-only mode `train` is None (not loaded) but
    num_train is still reported so ground-truth filtering stays correct."""
    with h5py.File(path, "r") as f:
        total_train = f["train"].shape[0]
        num_train = num_docs if (num_docs and num_docs < total_train) else total_train
        train = None if search_only else np.asarray(f["train"][:num_train], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"]) if "neighbors" in f else None
    if num_queries and num_queries < test.shape[0]:
        test = test[:num_queries]
        if gt is not None:
            gt = gt[:num_queries]
    return train, num_train, test, gt


def req(method, url, **kw):
    r = requests.request(method, url, timeout=kw.pop("timeout", 600), **kw)
    if not r.ok:
        raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text[:500]}")
    return r


PAGE_TOUCH_LOGGER = "org.opensearch.knn.index.codec.scorer.PageTouchTracker"
_PAGE_TOUCH_RE = re.compile(
    r"PageTouch .*dataUsedBytes=\[(\d+)\].*distinctVectors=\[(\d+)\].*recordSize=\[(\d+)\]"
    r".*32KB: dataReadBytes=\[(\d+)\].*?distinctPages=\[(\d+)\]"
    r".*8KB: dataReadBytes=\[(\d+)\].*?distinctPages=\[(\d+)\]"
)


def set_page_touch_logging(host, level):
    """Enable/disable the PageTouchTracker instrumentation at runtime (no node restart).
    level='DEBUG' turns tracking on; level=None removes the override."""
    body = {"persistent": {f"logger.{PAGE_TOUCH_LOGGER}": level}}
    req("PUT", f"{host}/_cluster/settings", headers={"Content-Type": "application/json"}, data=json.dumps(body))


def log_offset(path):
    """Current end-of-file byte offset (0 if the file is missing yet)."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def summarize_page_touch(path, start_offset, label):
    """Parse the PageTouch lines appended since start_offset and report READ AMPLIFICATION.
    Each line is one per-segment search event (== one query after a force-merge to 1 segment).
    Read amplification = bytes the device faulted in (whole 32KB pages) / bytes actually needed
    (visited vectors' records). Lower is better; 1.0x is perfect."""
    if not path:
        log("  page-touch: no --log-file given; skipping summary")
        return
    try:
        with open(path, "r", errors="ignore") as fh:
            fh.seek(start_offset)
            text = fh.read()
    except OSError as e:
        log(f"  page-touch: could not read {path}: {e}")
        return

    # Group by recordSize so the quantized store (112/144 B) and the full-precision .vec rescore reads
    # (dim*4, e.g. 3072/4096 B) report as SEPARATE streams instead of being summed together.
    # stream[recordSize] = [events, sum_used, sum_vecs, sum_read32, sum_pages32, sum_read8, sum_pages8]
    streams = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    for m in _PAGE_TOUCH_RE.finditer(text):
        used_b, vecs, rs, read32, pages32, read8, pages8 = (
            int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]), int(m[7])
        )
        s = streams[rs]
        s[0] += 1
        s[1] += used_b
        s[2] += vecs
        s[3] += read32
        s[4] += pages32
        s[5] += read8
        s[6] += pages8

    if not streams:
        log(f"  page-touch [{label}]: no PageTouch log lines found (is the logger at DEBUG? is --log-file correct?)")
        return

    mb = 1024 * 1024

    def kind(rs):
        if rs <= 512:
            return "quantized"
        return "full-precision(.vec)"

    log(f"  page-touch [{label}] — READ AMPLIFICATION (by stream):")
    for rs in sorted(streams):
        n, sum_used, sum_vecs, sum_read32, sum_pages32, sum_read8, sum_pages8 = streams[rs]
        log(f"    stream recordSize={rs}B [{kind(rs)}] — {n} search events:")
        log(f"      data used (vector records): {sum_used / mb:>9.1f} MB   (avg {sum_used / n / 1024:>8.1f} KB/query, avg {sum_vecs / n:>7.0f} vecs/query)")

        def report(sz_label, sum_read, sum_pages):
            amp = sum_read / sum_used if sum_used else 0.0
            util = 100.0 / amp if amp else 0.0
            vpp = sum_vecs / max(1, sum_pages)
            log(
                f"      [{sz_label}] read {sum_read / mb:>9.1f} MB  READ_AMP {amp:>7.1f}x  util {util:>5.2f}%  "
                f"avg pages/query {sum_pages / n:>8.1f}  vecs/page {vpp:>5.2f}"
            )

        report("32KB", sum_read32, sum_pages32)
        report(" 8KB", sum_read8, sum_pages8)


def delete_index(host, index):
    r = requests.delete(f"{host}/{index}", timeout=120)
    if r.status_code not in (200, 404):
        raise RuntimeError(f"delete {index} -> {r.status_code}: {r.text[:300]}")


def create_index(host, index, dim, field, space_type, shards, reordering_enabled):
    body = {
        "settings": {
            "index.knn": True,
            "number_of_shards": shards,
            "number_of_replicas": 0,
            "index.knn.advanced.reordering_enabled": reordering_enabled,
        },
        "mappings": {
            "properties": {
                field: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "space_type": space_type,
                    "mode": "on_disk",
                    "compression_level": "32x",
                }
            }
        },
    }
    req("PUT", f"{host}/{index}", headers={"Content-Type": "application/json"}, data=json.dumps(body))


def make_shuffle_maps(num_train, seed=42):
    """Deterministic decorrelation of doc id from vector content (fixed seed matches the sim).
    Returns (perm, inv_perm): doc _id j holds vector train[perm[j]]; a train index t lives at doc
    inv_perm[t]. Reproducible from (num_train, seed) alone, so --search-only reconstructs the same
    ground-truth remap without re-ingesting."""
    perm = np.random.default_rng(seed).permutation(num_train).astype(np.int64)
    inv_perm = np.empty(num_train, dtype=np.int64)
    inv_perm[perm] = np.arange(num_train, dtype=np.int64)
    return perm, inv_perm


def bulk_ingest(host, index, field, vectors, batch_size, perm=None):
    # perm decorrelates physical layout from vector space: doc _id i is assigned vector[perm[i]].
    # After a force-merge (which re-lays vectors in doc-id order) the physical ordinal == doc id, so
    # ordinal order stays uncorrelated with vector similarity — the realistic random-arrival case.
    n = vectors.shape[0]
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lines = []
        for i in range(start, end):
            src = int(perm[i]) if perm is not None else i
            lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
            lines.append(json.dumps({field: vectors[src].tolist()}))
        payload = "\n".join(lines) + "\n"
        r = req(
            "POST",
            f"{host}/_bulk",
            headers={"Content-Type": "application/x-ndjson"},
            data=payload,
        )
        if r.json().get("errors"):
            # Surface the first item error to fail fast.
            for item in r.json()["items"]:
                op = item.get("index", {})
                if op.get("error"):
                    raise RuntimeError(f"bulk error at doc {op.get('_id')}: {op['error']}")
        if end % (batch_size * 20) == 0 or end == n:
            rate = end / max(1e-9, time.time() - t0)
            log(f"  ingested {end}/{n} ({rate:.0f} docs/s)")


def refresh(host, index):
    req("POST", f"{host}/{index}/_refresh")


def count(host, index):
    return req("GET", f"{host}/{index}/_count").json()["count"]


def segment_count(host, index):
    stats = req("GET", f"{host}/{index}/_stats/segments").json()
    return stats["indices"][index]["primaries"]["segments"]["count"]


def knn_search(host, index, field, query, k, rescore, ef_search=None):
    knn_clause = {"vector": query.tolist(), "k": k, "rescore": rescore}
    # ef_search is a per-query knob passed via method_parameters; when None the index/engine default
    # is used. Higher ef_search visits more nodes (better recall, more pages touched).
    if ef_search is not None:
        knn_clause["method_parameters"] = {"ef_search": ef_search}
    body = {
        "size": k,
        "_source": False,
        "query": {"knn": {field: knn_clause}},
    }
    r = req("POST", f"{host}/{index}/_search", headers={"Content-Type": "application/json"}, data=json.dumps(body))
    return [hit["_id"] for hit in r.json()["hits"]["hits"]]


def evaluate_recall(host, index, field, queries, gt, k, num_docs, rescore, truth_map=None, ef_search=None):
    if gt is None:
        log("  no ground-truth neighbors in dataset; skipping recall")
        return None
    total = 0.0
    t0 = time.time()
    for q in range(queries.shape[0]):
        returned = set(knn_search(host, index, field, queries[q], k, rescore, ef_search=ef_search))
        # Only ground-truth ids that were actually ingested count (matters for subset runs). When
        # shuffle-ingest is active, a train index t was stored under doc id truth_map[t], so remap.
        truth_ids = [
            int(truth_map[int(x)]) if truth_map is not None else int(x)
            for x in gt[q]
            if int(x) < num_docs
        ][:k]
        truth = set(str(x) for x in truth_ids)
        total += len(returned & truth) / float(len(truth)) if truth else 0.0
        if (q + 1) % 200 == 0:
            log(f"  queried {q + 1}/{queries.shape[0]} ({(q + 1) / (time.time() - t0):.0f} q/s)")
    return total / queries.shape[0]


def run_config(
    host, index, field, train, queries, gt, k, space_type, shards, batch_size, reordering, rescore, search_only, num_docs,
    page_touch=False, log_file=None, ingest_perm=None, truth_map=None, ef_search=None,
):
    log(f"=== {index} (reordering_enabled={reordering}) ===")
    if search_only:
        log("  search-only: skipping delete/create/ingest, using the existing index")
    else:
        delete_index(host, index)
        create_index(host, index, train.shape[1], field, space_type, shards, reordering)
        shuffle_note = " (shuffle-ingest: doc id decorrelated from vector space)" if ingest_perm is not None else ""
        log(f"  ingesting {train.shape[0]} vectors (dim={train.shape[1]}){shuffle_note} ...")
        bulk_ingest(host, index, field, train, batch_size, perm=ingest_perm)
        refresh(host, index)
    log(f"  doc count: {count(host, index)}, segments: {segment_count(host, index)} (no force-merge)")
    log(f"  running {queries.shape[0]} queries (k={k}, rescore={rescore}, ef_search={ef_search or 'default'}) ...")
    # Mark the log position so the summary only counts lines this run appends.
    start_offset = log_offset(log_file) if page_touch else 0
    recall = evaluate_recall(host, index, field, queries, gt, k, num_docs, rescore, truth_map=truth_map, ef_search=ef_search)
    if page_touch:
        summarize_page_touch(log_file, start_offset, index)
    return recall


def main():
    p = argparse.ArgumentParser(description="Recall: reordered vs baseline Faiss SQ 1-bit (32x)")
    p.add_argument("--host", default="http://localhost:9200")
    p.add_argument("--dataset", default="scripts/documents-1m.hdf5")
    p.add_argument("--num-docs", type=int, default=0, help="base vectors to ingest (0 = all)")
    p.add_argument("--num-queries", type=int, default=1000, help="test queries (0 = all)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="per-query ef_search (method_parameters.ef_search); omit to use the index/engine default",
    )
    p.add_argument("--field", default="test_field")
    p.add_argument("--space-type", default="innerproduct", help="innerproduct | cosinesimil | l2")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--index-prefix", default="reorder-recall")
    p.add_argument("--rescore", action="store_true", help="enable full-precision rescore (default: off / rescore=false)")
    p.add_argument("--search-only", action="store_true", help="skip ingest; query the already-indexed data")
    p.add_argument(
        "--shuffle-ingest",
        action="store_true",
        help="decorrelate doc id from vector content (fixed seed 42) so the physical layout is NOT "
        "pre-sorted by vector space — models realistic random doc arrival, the worst case where "
        "reordering earns its keep. Must be paired with a force-merge to 1 segment before searching. "
        "Reproducible in --search-only (rebuilds the same map from count+seed).",
    )
    p.add_argument(
        "--page-touch",
        action="store_true",
        help="enable PageTouchTracker (via cluster-settings DEBUG) and print distinct-pages-per-query per run",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="path to the OpenSearch node log to scrape PageTouch lines from "
        "(e.g. build/testclusters/integTest-0/logs/<cluster>.log); required with --page-touch",
    )
    p.add_argument(
        "--only",
        choices=["both", "enabled", "disabled"],
        default="both",
        help="which index to run: 'enabled' = reordered only, 'disabled' = baseline only, 'both' (default)",
    )
    args = p.parse_args()

    log(f"loading {args.dataset} ...")
    train, num_train, test, gt = load_dataset(args.dataset, args.num_docs, args.num_queries, args.search_only)
    train_shape = "not loaded (search-only)" if train is None else train.shape
    log(
        f"train={train_shape} num_train={num_train} test={test.shape} "
        f"gt={None if gt is None else gt.shape} space={args.space_type} k={args.k} "
        f"rescore={args.rescore} search_only={args.search_only}"
    )

    if args.page_touch and not args.log_file:
        log("WARNING: --page-touch given without --log-file; per-query page summary will be skipped")

    # Shuffle-ingest: decorrelate doc id from vector content so the physical layout isn't pre-sorted
    # by vector space. Built from (num_train, seed) so it matches between an ingest run and a later
    # --search-only run. ingest_perm places vectors; truth_map (inverse) remaps ground truth.
    ingest_perm, truth_map = (None, None)
    if args.shuffle_ingest:
        ingest_perm, truth_map = make_shuffle_maps(num_train)
        log(f"shuffle-ingest: decorrelating doc id from vector content (num_train={num_train}, seed=42)")

    # Select which index(es) to run. 'enabled' = reordered only, 'disabled' = baseline only.
    if args.only == "enabled":
        toRun = [True]
    elif args.only == "disabled":
        toRun = [False]
    else:
        toRun = [False, True]

    if args.page_touch:
        log(f"page-touch: enabling {PAGE_TOUCH_LOGGER} at DEBUG via cluster settings")
        set_page_touch_logging(args.host, "DEBUG")

    results = {}
    try:
        for enabled in toRun:
            index = f"{args.index_prefix}-{'enabled' if enabled else 'disabled'}"
            results[enabled] = run_config(
                args.host, index, args.field, train, test, gt, args.k,
                args.space_type, args.shards, args.batch_size, enabled,
                args.rescore, args.search_only, num_train,
                page_touch=args.page_touch, log_file=args.log_file,
                ingest_perm=ingest_perm, truth_map=truth_map, ef_search=args.ef_search,
            )
    finally:
        if args.page_touch:
            log("page-touch: removing DEBUG logger override")
            set_page_touch_logging(args.host, None)

    print("\n" + "=" * 60)
    print(f"Recall@{args.k}  (docs={num_train}, queries={test.shape[0]}, space={args.space_type}, rescore={args.rescore})")
    print("-" * 60)
    base = results.get(False)
    reord = results.get(True)
    if False in results:
        print(f"  baseline (reordering disabled): {base if base is None else f'{base:.4f}'}")
    if True in results:
        print(f"  reordered (reordering enabled): {reord if reord is None else f'{reord:.4f}'}")
    if base is not None and reord is not None:
        print(f"  delta (reordered - baseline):   {reord - base:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
