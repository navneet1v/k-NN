#!/usr/bin/env python3
"""Cohere-1M benchmark for ClusterANN IVF on OpenSearch k-NN."""
import struct
import numpy as np
import requests
import time
import json
import sys

HOST = "http://localhost:9200"
INDEX = "clusterann-cohere-1m"
FIELD = "vector"
DIM = 768
BATCH_SIZE = 2000

def read_bin(filename):
    with open(filename, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        return np.fromfile(f, dtype=np.float32, count=count * dim).reshape(count, dim)

def read_groundtruth(filename):
    with open(filename, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        k = struct.unpack('i', f.read(4))[0]
        return np.fromfile(f, dtype=np.int32, count=count * k).reshape(count, k)

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
        if resp.json().get('errors'):
            for item in resp.json()['items']:
                if 'error' in item.get('index', {}):
                    print(f"Doc error: {item['index']['error']}")
                    sys.exit(1)
        if (i // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            print(f"  {i + len(batch):>7}/{len(base)} ({rate:.0f} vec/s)")
    elapsed = time.time() - t0
    print(f"Indexing done in {elapsed:.1f}s ({len(base)/elapsed:.0f} vec/s)")

def force_merge():
    print("\nForce merging to 1 segment...")
    t0 = time.time()
    resp = requests.post(f"{HOST}/{INDEX}/_forcemerge?max_num_segments=1&wait_for_completion=true")
    print(f"Force merge done in {time.time()-t0:.1f}s — {resp.status_code}")

def refresh():
    requests.post(f"{HOST}/{INDEX}/_refresh")

def search(query_vec, k):
    body = {
        "size": k,
        "query": {
            "knn": {
                FIELD: {
                    "vector": query_vec.tolist(),
                    "k": k
                }
            }
        }
    }
    resp = requests.post(f"{HOST}/{INDEX}/_search", json=body)
    return resp.json()

def benchmark_recall(queries, groundtruth, k=100, num_queries=100):
    print(f"\nBenchmarking recall@{k} on {num_queries} queries...")
    recalls = []
    latencies = []
    for i in range(num_queries):
        true_nn = set(groundtruth[i][:k].tolist())
        t0 = time.time()
        result = search(queries[i], k)
        latencies.append(time.time() - t0)
        returned = set(int(hit['_id']) for hit in result['hits']['hits'])
        recall = len(returned & true_nn) / k
        recalls.append(recall)
        if i % 20 == 0:
            print(f"  query {i:>4}: recall={recall:.3f}  latency={latencies[-1]*1000:.1f}ms")

    avg_recall = np.mean(recalls)
    p50 = np.percentile(latencies, 50) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    print(f"\n{'='*50}")
    print(f"Recall@{k}:  {avg_recall:.4f}  (min={min(recalls):.3f}, max={max(recalls):.3f})")
    print(f"Latency:    p50={p50:.1f}ms  p99={p99:.1f}ms  avg={np.mean(latencies)*1000:.1f}ms")
    print(f"{'='*50}")
    return avg_recall

if __name__ == '__main__':
    print("Loading Cohere-1M dataset...")
    base = read_bin('/Users/viktari/pysptag/data/cohere/base.bin')
    queries = read_bin('/Users/viktari/pysptag/data/cohere/query.bin')
    groundtruth = read_groundtruth('/Users/viktari/pysptag/data/cohere/groundtruth.bin')
    print(f"Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape}")
    print(f"Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape} (normalized)")

    # Delete old index
    requests.delete(f"{HOST}/{INDEX}")

    # Create index: Cohere uses inner_product, 2-bit (16x), 2x oversampling
    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                FIELD: {
                    "type": "knn_vector",
                    "dimension": DIM,
                    "compression_level": "16x",
                    "method": {
                        "name": "cluster",
                        "space_type": "innerproduct"
                    }
                }
            }
        }
    }
    resp = requests.put(f"{HOST}/{INDEX}", json=mapping)
    print(f"Index created: {resp.status_code} — {resp.json().get('acknowledged')}")
    if resp.status_code != 200:
        print(resp.text)
        sys.exit(1)

    bulk_index(base)
    refresh()
    force_merge()
    refresh()

    # Warmup
    print("\nWarmup (5 queries)...")
    for i in range(5):
        search(queries[i], 100)

    benchmark_recall(queries, groundtruth, k=100, num_queries=100)
