#!/usr/bin/env python3
"""100K Sequential Batch Dedup - batch=6000, matrix-multiply local dedup"""
import numpy as np, requests, json, time, sys
sys.stdout.reconfigure(line_buffering=True)

HOST = "http://localhost:9200"
INDEX = "video_dedup_100k"
DIM = 768
TOTAL = 100000
N_UNIQUE = 95000
N_DUPS = 5000
COSINE_THRESHOLD = 0.95
MIN_SCORE = 1.0 + COSINE_THRESHOLD  # 1.95
BATCH_SIZE = 6000
MSEARCH_CHUNK = 1000

print(f"=== 100K | batch={BATCH_SIZE} | local dedup (matrix) + _msearch radial ===")

print("Generating data...")
np.random.seed(42)
unique = np.random.randn(N_UNIQUE, DIM).astype('float32')
unique = unique / np.linalg.norm(unique, axis=1, keepdims=True)
dup_src = np.random.choice(N_UNIQUE, N_DUPS)
dups = unique[dup_src] + np.random.randn(N_DUPS, DIM).astype('float32') * 0.01
dups = dups / np.linalg.norm(dups, axis=1, keepdims=True)
all_vecs = np.vstack([unique, dups])[np.random.permutation(TOTAL)]
print(f"Ready: {TOTAL} vecs ({N_UNIQUE} unique + {N_DUPS} dups)")

requests.delete(f"{HOST}/{INDEX}")
time.sleep(1)
requests.put(f"{HOST}/{INDEX}", json={
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0,
                           "refresh_interval": "1s", "knn": True,
                           "knn.algo_param.ef_search": 256}},
    "mappings": {"properties": {"embedding": {
        "type": "knn_vector", "dimension": DIM, "data_type": "float",
        "method": {"name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                   "parameters": {"m": 16, "ef_construction": 128}}}}}
})
time.sleep(2)
print("Index created. Starting...\n")

indexed = 0; discarded = 0; t0 = time.time()
num_batches = (TOTAL + BATCH_SIZE - 1) // BATCH_SIZE

for bn in range(num_batches):
    s = bn * BATCH_SIZE
    batch = all_vecs[s:min(s+BATCH_SIZE, TOTAL)]
    bs = len(batch)

    # 1) Local dedup — matrix multiply approach
    t1 = time.time()
    sim_matrix = batch @ batch.T
    np.fill_diagonal(sim_matrix, 0)
    to_keep = np.ones(bs, dtype=bool)
    for i in range(1, bs):
        if to_keep[i]:
            if sim_matrix[i, :i][to_keep[:i]].max() >= COSINE_THRESHOLD:
                to_keep[i] = False
                discarded += 1
    batch_unique = batch[to_keep]
    local_time = time.time() - t1

    # 2) _msearch in chunks (radial search)
    t2 = time.time()
    survs = []
    for cs in range(0, len(batch_unique), MSEARCH_CHUNK):
        chunk = batch_unique[cs:cs+MSEARCH_CHUNK]
        lines = []
        for v in chunk:
            lines.append(json.dumps({"index": INDEX}))
            lines.append(json.dumps({"size": 1, "query": {"knn": {"embedding": {"vector": v.tolist(), "min_score": MIN_SCORE}}}}))
        r = requests.post(f"{HOST}/_msearch", data="\n".join(lines)+"\n",
                          headers={"Content-Type": "application/x-ndjson"})
        if r.status_code != 200:
            print(f"FAIL: {r.status_code}"); break
        for i, res in enumerate(r.json()["responses"]):
            if res.get("hits", {}).get("total", {}).get("value", 0) == 0:
                survs.append(chunk[i])
            else:
                discarded += 1
    search_time = time.time() - t2

    # 3) _bulk index
    t3 = time.time()
    if survs:
        bl = []
        for v in survs:
            bl.append(json.dumps({"index": {"_index": INDEX}}))
            bl.append(json.dumps({"embedding": v.tolist()}))
            indexed += 1
        requests.post(f"{HOST}/_bulk", data="\n".join(bl)+"\n",
                      headers={"Content-Type": "application/x-ndjson"})
    bulk_time = time.time() - t3

    # 4) Wait for refresh
    time.sleep(1)

    elapsed = time.time() - t0
    print(f"Batch {bn+1:2d}/{num_batches} | Local:{local_time:.2f}s Search:{search_time:.1f}s Bulk:{bulk_time:.1f}s | "
          f"Idx={indexed:6d} Disc={discarded:4d} | Elapsed:{elapsed:.0f}s")

total_time = time.time() - t0
print(f"\nDONE: Indexed={indexed}, Discarded={discarded}, Time={total_time:.0f}s")
print(f"Expected: ~{N_UNIQUE} indexed, ~{N_DUPS} discarded")
print(f"Missed: {indexed - N_UNIQUE} (0=perfect)")
