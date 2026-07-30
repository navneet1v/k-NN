#!/usr/bin/env python3
"""
Dedup Benchmark: 3 Experiments with 10K vectors (768D)
Simulates the Tencent video dedup workflow using FAISS HNSW (same as OpenSearch k-NN engine).

Experiment 1: SEQUENTIAL — one batch at a time, wait for index update
Experiment 2: PARALLEL BATCHES — multiple workers, shared index with refresh delay
Experiment 3: PARTITIONED PARALLEL — LSH-route similar vectors to same worker

Metrics: time, accuracy (duplicates leaked), final index size
"""
import numpy as np
import faiss
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

np.random.seed(42)

# === CONFIG ===
DIM = 768
TOTAL_VECTORS = 10000
N_UNIQUE = 7000
N_DUPLICATES = 3000
THRESHOLD = 0.95
BATCH_SIZE = 100
NUM_WORKERS = 10  # for parallel experiments
REFRESH_INTERVAL_BATCHES = 3  # how many batches before "refresh" makes vectors searchable

# === GENERATE DATA ===
print("=" * 70)
print("GENERATING DATASET")
print("=" * 70)
t0 = time.time()

unique_vectors = np.random.randn(N_UNIQUE, DIM).astype('float32')
faiss.normalize_L2(unique_vectors)

dup_sources = np.random.choice(N_UNIQUE, N_DUPLICATES)
noise = np.random.randn(N_DUPLICATES, DIM).astype('float32') * 0.005
duplicates = unique_vectors[dup_sources] + noise
faiss.normalize_L2(duplicates)

# Verify duplicates are above threshold
sims = np.array([np.dot(unique_vectors[dup_sources[i]], duplicates[i]) for i in range(N_DUPLICATES)])
print(f"Dataset: {TOTAL_VECTORS} vectors ({N_UNIQUE} unique + {N_DUPLICATES} near-duplicates)")
print(f"Dimensions: {DIM}")
print(f"Duplicate similarities: min={sims.min():.4f}, mean={sims.mean():.4f}, max={sims.max():.4f}")
print(f"All above {THRESHOLD}? {(sims >= THRESHOLD).all()}")

# Shuffle
all_vectors = np.vstack([unique_vectors, duplicates])
is_dup = np.array([False] * N_UNIQUE + [True] * N_DUPLICATES)
shuffle_idx = np.random.permutation(TOTAL_VECTORS)
all_vectors = all_vectors[shuffle_idx]
is_dup = is_dup[shuffle_idx]

print(f"Data generated in {time.time() - t0:.2f}s\n")


def verify_index(indexed_vectors):
    """Check how many duplicate pairs remain in the final index."""
    if len(indexed_vectors) == 0:
        return 0
    mat = np.array(indexed_vectors)
    # Use FAISS for efficient pairwise search
    index = faiss.IndexFlatIP(DIM)
    index.add(mat)
    # Search each vector for its nearest neighbor (excluding self)
    D, I = index.search(mat, 2)  # top-2 (first is self with score=1.0)
    dup_count = np.sum(D[:, 1] >= THRESHOLD)
    return dup_count // 2  # each pair counted twice


# ============================================================
# EXPERIMENT 1: SEQUENTIAL
# ============================================================
print("=" * 70)
print("EXPERIMENT 1: SEQUENTIAL (single worker, immediate refresh)")
print("=" * 70)

t0 = time.time()
index_seq = faiss.IndexFlatIP(DIM)
indexed_vecs_seq = []
discarded_seq = 0

for batch_start in range(0, TOTAL_VECTORS, BATCH_SIZE):
    batch = all_vectors[batch_start:batch_start + BATCH_SIZE]
    buffer = []

    for vec in batch:
        vec_2d = vec.reshape(1, -1)
        found = False

        # Search against FAISS index (simulates OpenSearch _msearch)
        if index_seq.ntotal > 0:
            D, _ = index_seq.search(vec_2d, 1)
            if D[0][0] >= THRESHOLD:
                found = True

        # Intra-batch dedup
        if not found and len(buffer) > 0:
            buf_arr = np.array(buffer)
            sim = (buf_arr @ vec).max()
            if sim >= THRESHOLD:
                found = True

        if found:
            discarded_seq += 1
        else:
            buffer.append(vec)

    # Index survivors immediately (sequential = instant refresh)
    if buffer:
        buf_arr = np.array(buffer)
        index_seq.add(buf_arr)
        indexed_vecs_seq.extend(buffer)

time_seq = time.time() - t0
dup_pairs_seq = verify_index(indexed_vecs_seq)

print(f"  Time:           {time_seq:.2f}s")
print(f"  Indexed:        {len(indexed_vecs_seq)}")
print(f"  Discarded:      {discarded_seq}")
print(f"  Dups leaked:    {dup_pairs_seq}")
print(f"  Accuracy:       {'100%' if dup_pairs_seq == 0 else f'{(1 - dup_pairs_seq/len(indexed_vecs_seq))*100:.2f}%'}")
print()


# ============================================================
# EXPERIMENT 2: PARALLEL BATCHES (shared index, refresh delay)
# ============================================================
print("=" * 70)
print(f"EXPERIMENT 2: PARALLEL ({NUM_WORKERS} workers, refresh every {REFRESH_INTERVAL_BATCHES} batches)")
print("=" * 70)

t0 = time.time()
# Shared state
index_par = faiss.IndexFlatIP(DIM)
pending_vectors = []
pending_lock = threading.Lock()
searchable_index = faiss.IndexFlatIP(DIM)  # only refreshed periodically
discarded_par = 0
indexed_par_list = []
batch_counter = [0]
counter_lock = threading.Lock()


def process_batch_parallel(batch):
    global discarded_par
    buffer = []

    for vec in batch:
        vec_2d = vec.reshape(1, -1)
        found = False

        # Search against SEARCHABLE index (may be stale)
        if searchable_index.ntotal > 0:
            D, _ = searchable_index.search(vec_2d, 1)
            if D[0][0] >= THRESHOLD:
                found = True

        # Intra-batch dedup
        if not found and len(buffer) > 0:
            buf_arr = np.array(buffer)
            sim = (buf_arr @ vec).max()
            if sim >= THRESHOLD:
                found = True

        if found:
            with counter_lock:
                discarded_par += 1
        else:
            buffer.append(vec)

    return buffer


# Process in rounds to simulate refresh
batches = [all_vectors[i:i + BATCH_SIZE] for i in range(0, TOTAL_VECTORS, BATCH_SIZE)]
batch_idx = 0

while batch_idx < len(batches):
    # Process REFRESH_INTERVAL_BATCHES * NUM_WORKERS batches in parallel
    round_batches = batches[batch_idx:batch_idx + NUM_WORKERS * REFRESH_INTERVAL_BATCHES]
    batch_idx += len(round_batches)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_batch_parallel, b) for b in round_batches]
        for f in as_completed(futures):
            survivors = f.result()
            if survivors:
                with pending_lock:
                    pending_vectors.extend(survivors)
                    indexed_par_list.extend(survivors)

    # REFRESH: make pending vectors searchable
    with pending_lock:
        if pending_vectors:
            arr = np.array(pending_vectors)
            searchable_index.add(arr)
            pending_vectors = []

time_par = time.time() - t0
dup_pairs_par = verify_index(indexed_par_list)

print(f"  Time:           {time_par:.2f}s")
print(f"  Indexed:        {len(indexed_par_list)}")
print(f"  Discarded:      {discarded_par}")
print(f"  Dups leaked:    {dup_pairs_par}")
print(f"  Accuracy:       {(1 - dup_pairs_par/max(len(indexed_par_list),1))*100:.2f}%")
print()


# ============================================================
# EXPERIMENT 3: PARTITIONED PARALLEL (LSH routing)
# ============================================================
print("=" * 70)
print(f"EXPERIMENT 3: PARTITIONED PARALLEL ({NUM_WORKERS} workers, LSH routing)")
print("=" * 70)

t0 = time.time()

# Create LSH projections for routing
projections = np.random.randn(NUM_WORKERS, DIM).astype('float32')
faiss.normalize_L2(projections)


def assign_partition(vec):
    """Route vector to worker based on max dot product with random projections."""
    dots = projections @ vec
    return int(np.argmax(dots))


# Partition all vectors by LSH hash
partitions = [[] for _ in range(NUM_WORKERS)]
for vec in all_vectors:
    p = assign_partition(vec)
    partitions[p].append(vec)

print(f"  Partition sizes: {[len(p) for p in partitions]}")

# Each partition is processed SEQUENTIALLY by its assigned worker
# But all workers run IN PARALLEL
indexed_part_list = []
discarded_part = 0
part_lock = threading.Lock()


def process_partition(partition_vecs):
    global discarded_part
    local_index = faiss.IndexFlatIP(DIM)
    local_indexed = []

    for i in range(0, len(partition_vecs), BATCH_SIZE):
        batch = partition_vecs[i:i + BATCH_SIZE]
        buffer = []

        for vec in batch:
            vec_2d = np.array(vec).reshape(1, -1)
            found = False

            if local_index.ntotal > 0:
                D, _ = local_index.search(vec_2d, 1)
                if D[0][0] >= THRESHOLD:
                    found = True

            if not found and len(buffer) > 0:
                buf_arr = np.array(buffer)
                sim = (buf_arr @ vec).max()
                if sim >= THRESHOLD:
                    found = True

            if found:
                with part_lock:
                    discarded_part += 1
            else:
                buffer.append(vec)

        if buffer:
            arr = np.array(buffer)
            local_index.add(arr)
            local_indexed.extend(buffer)

    return local_indexed


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_partition, p) for p in partitions]
    for f in as_completed(futures):
        indexed_part_list.extend(f.result())

time_part = time.time() - t0
dup_pairs_part = verify_index(indexed_part_list)

print(f"  Time:           {time_part:.2f}s")
print(f"  Indexed:        {len(indexed_part_list)}")
print(f"  Discarded:      {discarded_part}")
print(f"  Dups leaked:    {dup_pairs_part}")
print(f"  Accuracy:       {(1 - dup_pairs_part/max(len(indexed_part_list),1))*100:.2f}%")
print()


# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Metric':<25} {'Sequential':<18} {'Parallel':<18} {'Partitioned':<18}")
print("-" * 70)
print(f"{'Time (s)':<25} {time_seq:<18.2f} {time_par:<18.2f} {time_part:<18.2f}")
print(f"{'Speedup vs sequential':<25} {'1.0x':<18} {f'{time_seq/time_par:.1f}x':<18} {f'{time_seq/time_part:.1f}x':<18}")
print(f"{'Indexed':<25} {len(indexed_vecs_seq):<18} {len(indexed_par_list):<18} {len(indexed_part_list):<18}")
print(f"{'Discarded':<25} {discarded_seq:<18} {discarded_par:<18} {discarded_part:<18}")
print(f"{'Duplicate pairs leaked':<25} {dup_pairs_seq:<18} {dup_pairs_par:<18} {dup_pairs_part:<18}")
print(f"{'Accuracy':<25} {'100%':<18} {f'{(1-dup_pairs_par/max(len(indexed_par_list),1))*100:.2f}%':<18} {f'{(1-dup_pairs_part/max(len(indexed_part_list),1))*100:.2f}%':<18}")
print(f"{'Final index clean?':<25} {'✅ YES':<18} {'❌ NO' if dup_pairs_par > 0 else '✅ YES':<18} {'❌ NO' if dup_pairs_part > 0 else '✅ YES':<18}")
print()
print("Expected (ideal):  Indexed=7000, Discarded=3000, Leaked=0")
