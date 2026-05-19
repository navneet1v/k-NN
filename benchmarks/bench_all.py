#!/usr/bin/env python3
"""ClusterANN benchmark — runs all datasets with configurable parameters."""
import struct, numpy as np, requests, time, json, sys, os

HOST = "http://localhost:9200"
BATCH_SIZE = 5000
NUM_QUERIES = 10000

DATA_DIR = "/Users/viktari/pysptag/data"

DATASETS = {
    "sift": {"dim": 128, "space": "l2", "compression": "16x", "format": "bin"},
    "cohere": {"dim": 768, "space": "innerproduct", "compression": "16x", "format": "bin"},
    "gist": {"dim": 960, "space": "l2", "compression": "16x", "format": "bin"},
    "glove": {"dim": 200, "space": "cosinesimil", "compression": "16x", "format": "hdf5", "file": "glove-200-angular.hdf5"},
    "flicker": {"dim": 512, "space": "innerproduct", "compression": "16x", "format": "hdf5", "file": "FlickrImagesTextQueries.hdf5"},
    "mbread": {"dim": 1024, "space": "innerproduct", "compression": "16x", "format": "hdf5", "file": "mbread_msmarco.hdf5"},
    "mpnetMarco": {"dim": 768, "space": "innerproduct", "compression": "16x", "format": "hdf5", "file": "mpnet_marco.hdf5"},
    "tasb": {"dim": 768, "space": "innerproduct", "compression": "16x", "format": "hdf5", "file": "marco_tasb.hdf5"},
}

OVERSAMPLE_FACTORS = [2.0, 3.0, 5.0]

def read_bin(f):
    with open(f, 'rb') as fh:
        c = struct.unpack('i', fh.read(4))[0]
        d = struct.unpack('i', fh.read(4))[0]
        return np.fromfile(fh, dtype=np.float32, count=c * d).reshape(c, d)

def read_gt(f):
    with open(f, 'rb') as fh:
        c = struct.unpack('i', fh.read(4))[0]
        k = struct.unpack('i', fh.read(4))[0]
        return np.fromfile(fh, dtype=np.int32, count=c * k).reshape(c, k)

def load_dataset(name, cfg):
    data_path = f"{DATA_DIR}/{name}"
    if cfg.get("format") == "hdf5":
        import h5py
        f = h5py.File(f"{data_path}/{cfg['file']}", 'r')
        base = np.array(f['train'], dtype=np.float32)
        queries = np.array(f['test'], dtype=np.float32)
        gt = np.array(f['neighbors'], dtype=np.int32)
        f.close()
    else:
        base = read_bin(f"{data_path}/base.bin")
        queries = read_bin(f"{data_path}/query.bin")
        gt = read_gt(f"{data_path}/groundtruth.bin")
    return base, queries, gt

def run_benchmark(name, cfg, force_merge=True):
    INDEX = f"clusterann-{name}"
    FIELD = "vector"
    dim = cfg["dim"]
    data_path = f"{DATA_DIR}/{name}"

    if not os.path.exists(data_path):
        print(f"[{name}] SKIP — no data at {data_path}")
        return

    print(f"\n{'='*70}")
    print(f"[{name}] Loading {data_path}...")
    base, queries, gt = load_dataset(name, cfg)
    print(f"[{name}] Base: {base.shape}, Queries: {queries.shape}, GT: {gt.shape}")

    # Create index
    requests.delete(f"{HOST}/{INDEX}")
    mapping = {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {"properties": {FIELD: {
            "type": "knn_vector", "dimension": dim,
            "compression_level": cfg["compression"],
            "method": {"name": "cluster", "space_type": cfg["space"]}
        }}}
    }
    resp = requests.put(f"{HOST}/{INDEX}", json=mapping)
    if resp.status_code != 200:
        print(f"[{name}] FAILED to create index: {resp.text[:200]}")
        return
    print(f"[{name}] Index created")

    # Index vectors
    t0 = time.time()
    for i in range(0, len(base), BATCH_SIZE):
        batch = base[i:i + BATCH_SIZE]
        lines = []
        for j, vec in enumerate(batch):
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(i + j)}}))
            lines.append(json.dumps({FIELD: vec.tolist()}))
        resp = requests.post(f"{HOST}/_bulk", data='\n'.join(lines) + '\n',
                             headers={'Content-Type': 'application/x-ndjson'})
        if resp.status_code != 200 or resp.json().get('errors'):
            print(f"[{name}] Bulk error at {i}")
            return
        if (i // BATCH_SIZE) % 100 == 0 and i > 0:
            print(f"  {i}/{len(base)} ({i / (time.time() - t0):.0f} vec/s)")
    index_time = time.time() - t0
    print(f"[{name}] Indexed in {index_time:.0f}s ({len(base) / index_time:.0f} vec/s)")

    requests.post(f"{HOST}/{INDEX}/_refresh")

    if force_merge:
        print(f"[{name}] Force merging...")
        t0 = time.time()
        requests.post(f"{HOST}/{INDEX}/_forcemerge?max_num_segments=1&wait_for_completion=true")
        print(f"[{name}] Merge done in {time.time() - t0:.0f}s")
        requests.post(f"{HOST}/{INDEX}/_refresh")

    # Get segment count
    segs = requests.get(f"{HOST}/{INDEX}/_segments").json()
    num_segs = len(segs['indices'][INDEX]['shards']['0'][0]['segments'])
    print(f"[{name}] Segments: {num_segs}")

    # Warmup
    for i in range(10):
        requests.post(f"{HOST}/{INDEX}/_search",
                      json={'size': 100, 'query': {'knn': {FIELD: {'vector': queries[i].tolist(), 'k': 100}}}})

    # Benchmark each oversample factor
    num_q = min(NUM_QUERIES, len(queries))
    for osf in OVERSAMPLE_FACTORS:
        recalls, latencies = [], []
        for i in range(num_q):
            body = {'size': 100, 'query': {'knn': {FIELD: {
                'vector': queries[i].tolist(), 'k': 100,
                'rescore': {'oversample_factor': osf}
            }}}}
            t0 = time.time()
            resp = requests.post(f"{HOST}/{INDEX}/_search", json=body)
            latencies.append(time.time() - t0)
            result = resp.json()
            if 'hits' not in result or 'hits' not in result['hits']:
                continue
            returned = set(int(hit['_id']) for hit in result['hits']['hits'])
            recalls.append(len(returned & set(gt[i][:100].tolist())) / 100)
            if i % 2000 == 0 and i > 0:
                print(f"  [{name}] osf={osf:.0f}x q{i}: recall={np.mean(recalls):.4f}")

        if recalls:
            print(f"\n  [{name}] osf={osf:.0f}x | Recall@100: {np.mean(recalls):.4f} "
                  f"(min={min(recalls):.3f}) | p50={np.percentile(latencies, 50) * 1000:.0f}ms "
                  f"p99={np.percentile(latencies, 99) * 1000:.0f}ms")

    # Cleanup
    del base, queries, gt

if __name__ == '__main__':
    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    force_merge = "--no-force-merge" not in sys.argv

    print(f"Datasets: {datasets_to_run}, force_merge={force_merge}")
    for name in datasets_to_run:
        if name.startswith("--"):
            continue
        if name in DATASETS:
            run_benchmark(name, DATASETS[name], force_merge=force_merge)
        else:
            print(f"Unknown dataset: {name}")

    print("\n\nAll done!")
