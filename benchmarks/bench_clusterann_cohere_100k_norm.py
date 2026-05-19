#!/usr/bin/env python3
"""Cohere-100K normalized benchmark for ClusterANN."""
import numpy as np
import h5py
import requests
import time
import json
import sys

HOST = "http://localhost:9200"
INDEX = "clusterann-cohere-100k-norm"
FIELD = "vector"
DIM = 768
BATCH_SIZE = 2000

def bulk_index(base):
    print(f"\nIndexing {len(base)} vectors in batches of {BATCH_SIZE}...")
    t0 = time.time()
    for i in range(0, len(base), BATCH_SIZE):
        batch = base[i:i+BATCH_SIZE]
        lines = []
        for j, vec in enumerate(batch):
            doc_id = i + j
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(doc_id)}}))
            lines.append(json.dumps({FIELD: vec.tolist()}))
        body = '\n'.join(lines) + '\n'
        resp = requests.post(f"{HOST}/_bulk", data=body,
                           headers={'Content-Type': 'application/x-ndjson'})
        if resp.status_code != 200:
            print(f"Bulk error at {i}: {resp.text[:200]}")
            sys.exit(1)
        if (i // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            print(f"  {i + len(batch):>7}/{len(base)} ({rate:.0f} vec/s)")
    print(f"Indexing done in {time.time()-t0:.1f}s")

def force_merge():
    print("\nForce merging...")
    t0 = time.time()
    requests.post(f"{HOST}/{INDEX}/_forcemerge?max_num_segments=1&wait_for_completion=true")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    print("Loading Cohere-100K normalized...")
    f = h5py.File('/Users/viktari/cohere-100k-normalized.hdf5', 'r')
    base = np.array(f['train'])
    queries = np.array(f['test'])
    groundtruth = np.array(f['neighbors'])
    f.close()
    print(f"Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape}")

    requests.delete(f"{HOST}/{INDEX}")

    # Normalized vectors + IP = cosine similarity
    mapping = {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {"properties": {FIELD: {
            "type": "knn_vector",
            "dimension": DIM,
            "compression_level": "16x",
            "method": {"name": "cluster", "space_type": "innerproduct"}
        }}}
    }
    resp = requests.put(f"{HOST}/{INDEX}", json=mapping)
    print(f"Index created: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text); sys.exit(1)

    bulk_index(base)
    requests.post(f"{HOST}/{INDEX}/_refresh")
    force_merge()
    requests.post(f"{HOST}/{INDEX}/_refresh")

    # Warmup
    for i in range(5):
        body = {'size': 100, 'query': {'knn': {FIELD: {'vector': queries[i].tolist(), 'k': 100}}}}
        requests.post(f"{HOST}/{INDEX}/_search", json=body)

    # Benchmark
    print(f"\nBenchmarking recall@100 on 100 queries...")
    recalls = []
    latencies = []
    for i in range(100):
        true_nn = set(groundtruth[i][:100].tolist())
        body = {'size': 100, 'query': {'knn': {FIELD: {'vector': queries[i].tolist(), 'k': 100}}}}
        t0 = time.time()
        resp = requests.post(f"{HOST}/{INDEX}/_search", json=body)
        latencies.append(time.time() - t0)
        result = resp.json()
        returned = set(int(hit['_id']) for hit in result['hits']['hits'])
        recall = len(returned & true_nn) / 100
        recalls.append(recall)
        if i % 20 == 0:
            print(f"  query {i:>4}: recall={recall:.3f}  latency={latencies[-1]*1000:.1f}ms")

    print(f"\n{'='*60}")
    print(f"Cohere-100K normalized | IP | 2-bit (16x) | 2*sqrt nprobe")
    print(f"Recall@100:  {np.mean(recalls):.4f}  (min={min(recalls):.3f}, max={max(recalls):.3f})")
    print(f"Latency:     p50={np.percentile(latencies,50)*1000:.1f}ms  p99={np.percentile(latencies,99)*1000:.1f}ms")
    print(f"{'='*60}")
