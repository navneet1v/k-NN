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
import time

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


def bulk_ingest(host, index, field, vectors, batch_size):
    n = vectors.shape[0]
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lines = []
        for i in range(start, end):
            lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
            lines.append(json.dumps({field: vectors[i].tolist()}))
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


def knn_search(host, index, field, query, k, rescore):
    knn_clause = {"vector": query.tolist(), "k": k, "rescore": rescore}
    body = {
        "size": k,
        "_source": False,
        "query": {"knn": {field: knn_clause}},
    }
    r = req("POST", f"{host}/{index}/_search", headers={"Content-Type": "application/json"}, data=json.dumps(body))
    return [hit["_id"] for hit in r.json()["hits"]["hits"]]


def evaluate_recall(host, index, field, queries, gt, k, num_docs, rescore):
    if gt is None:
        log("  no ground-truth neighbors in dataset; skipping recall")
        return None
    total = 0.0
    t0 = time.time()
    for q in range(queries.shape[0]):
        returned = set(knn_search(host, index, field, queries[q], k, rescore))
        # Only ground-truth ids that were actually ingested count (matters for subset runs).
        truth_ids = [int(x) for x in gt[q] if int(x) < num_docs][:k]
        truth = set(str(x) for x in truth_ids)
        total += len(returned & truth) / float(len(truth)) if truth else 0.0
        if (q + 1) % 200 == 0:
            log(f"  queried {q + 1}/{queries.shape[0]} ({(q + 1) / (time.time() - t0):.0f} q/s)")
    return total / queries.shape[0]


def run_config(host, index, field, train, queries, gt, k, space_type, shards, batch_size, reordering, rescore, search_only, num_docs):
    log(f"=== {index} (reordering_enabled={reordering}) ===")
    if search_only:
        log("  search-only: skipping delete/create/ingest, using the existing index")
    else:
        delete_index(host, index)
        create_index(host, index, train.shape[1], field, space_type, shards, reordering)
        log(f"  ingesting {train.shape[0]} vectors (dim={train.shape[1]}) ...")
        bulk_ingest(host, index, field, train, batch_size)
        refresh(host, index)
    log(f"  doc count: {count(host, index)}, segments: {segment_count(host, index)} (no force-merge)")
    log(f"  running {queries.shape[0]} queries (k={k}, rescore={rescore}) ...")
    recall = evaluate_recall(host, index, field, queries, gt, k, num_docs, rescore)
    return recall


def main():
    p = argparse.ArgumentParser(description="Recall: reordered vs baseline Faiss SQ 1-bit (32x)")
    p.add_argument("--host", default="http://localhost:9200")
    p.add_argument("--dataset", default="scripts/documents-1m.hdf5")
    p.add_argument("--num-docs", type=int, default=0, help="base vectors to ingest (0 = all)")
    p.add_argument("--num-queries", type=int, default=1000, help="test queries (0 = all)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--field", default="test_field")
    p.add_argument("--space-type", default="innerproduct", help="innerproduct | cosinesimil | l2")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--index-prefix", default="reorder-recall")
    p.add_argument("--rescore", action="store_true", help="enable full-precision rescore (default: off / rescore=false)")
    p.add_argument("--search-only", action="store_true", help="skip ingest; query the already-indexed data")
    args = p.parse_args()

    log(f"loading {args.dataset} ...")
    train, num_train, test, gt = load_dataset(args.dataset, args.num_docs, args.num_queries, args.search_only)
    train_shape = "not loaded (search-only)" if train is None else train.shape
    log(
        f"train={train_shape} num_train={num_train} test={test.shape} "
        f"gt={None if gt is None else gt.shape} space={args.space_type} k={args.k} "
        f"rescore={args.rescore} search_only={args.search_only}"
    )

    results = {}
    for enabled in (False, True):
        index = f"{args.index_prefix}-{'enabled' if enabled else 'disabled'}"
        results[enabled] = run_config(
            args.host, index, args.field, train, test, gt, args.k,
            args.space_type, args.shards, args.batch_size, enabled,
            args.rescore, args.search_only, num_train,
        )

    print("\n" + "=" * 60)
    print(f"Recall@{args.k}  (docs={num_train}, queries={test.shape[0]}, space={args.space_type}, rescore={args.rescore})")
    print("-" * 60)
    base = results[False]
    reord = results[True]
    print(f"  baseline (reordering disabled): {base if base is None else f'{base:.4f}'}")
    print(f"  reordered (reordering enabled): {reord if reord is None else f'{reord:.4f}'}")
    if base is not None and reord is not None:
        print(f"  delta (reordered - baseline):   {reord - base:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
