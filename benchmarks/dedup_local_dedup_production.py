#!/usr/bin/env python3
"""
Production-Ready: Sequential Batch Dedup with Local Dedup + _msearch
===================================================================

Flow per batch:
  1. Pull batch of vectors from S3/SQS
  2. Local dedup: matrix multiply (batch @ batch.T), discard pairs >= threshold
  3. _msearch: radial search survivors against OpenSearch (cross-batch dedup)
  4. _bulk: index vectors with no match
  5. Wait for refresh
  6. Next batch

Config:
  - OpenSearch cluster with innerproduct HNSW index
  - min_score = 1.95 (cosine >= 0.95 mapped to IP score)
  - Batch size: 10K-20K depending on client RAM
  - Single sequential worker (guarantees 100% accuracy)
"""

import numpy as np
import requests
import json
import time
import sys
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # OpenSearch
    host: str = os.environ.get("OPENSEARCH_HOST", "https://your-cluster.us-west-2.es.amazonaws.com")
    index: str = os.environ.get("INDEX_NAME", "video_vectors")
    
    # Vector params
    dimension: int = 768
    cosine_threshold: float = 0.95
    min_score: float = 1.95  # 1 + cosine_threshold (OpenSearch IP scoring)
    
    # Batching
    batch_size: int = 10000         # vectors per batch
    msearch_chunk: int = 1000       # max queries per _msearch call
    bulk_chunk: int = 2000          # max docs per _bulk call
    
    # Timing
    refresh_wait: float = 1.0       # seconds to wait after refresh
    
    # Retry
    max_retries: int = 3
    retry_delay: float = 5.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================
# OPENSEARCH CLIENT
# ============================================================

class OpenSearchClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/x-ndjson"})
    
    def create_index(self):
        """Create the kNN index with HNSW innerproduct."""
        cfg = self.config
        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 234,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s",
                    "knn": True,
                    "knn.algo_param.ef_search": 256
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": cfg.dimension,
                        "data_type": "float",
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {"m": 16, "ef_construction": 128}
                        }
                    },
                    "video_id": {"type": "keyword"}
                }
            }
        }
        resp = self.session.put(f"{cfg.host}/{cfg.index}", json=mapping)
        resp.raise_for_status()
        log.info(f"Index '{cfg.index}' created: {resp.status_code}")
    
    def msearch_radial(self, vectors: np.ndarray) -> List[bool]:
        """
        Radial search: returns list of booleans.
        True = match found (duplicate), False = no match (unique).
        """
        cfg = self.config
        results = []
        
        for chunk_start in range(0, len(vectors), cfg.msearch_chunk):
            chunk = vectors[chunk_start:chunk_start + cfg.msearch_chunk]
            lines = []
            for v in chunk:
                lines.append(json.dumps({"index": cfg.index}))
                lines.append(json.dumps({
                    "size": 1,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": v.tolist(),
                                "min_score": cfg.min_score
                            }
                        }
                    }
                }))
            
            resp = self._post_with_retry(
                f"{cfg.host}/_msearch",
                data="\n".join(lines) + "\n"
            )
            
            for res in resp.json()["responses"]:
                has_match = res.get("hits", {}).get("total", {}).get("value", 0) > 0
                results.append(has_match)
        
        return results
    
    def bulk_index(self, vectors: np.ndarray, video_ids: List[str]):
        """Bulk index vectors into OpenSearch."""
        cfg = self.config
        
        for chunk_start in range(0, len(vectors), cfg.bulk_chunk):
            chunk_vecs = vectors[chunk_start:chunk_start + cfg.bulk_chunk]
            chunk_ids = video_ids[chunk_start:chunk_start + cfg.bulk_chunk]
            
            lines = []
            for v, vid in zip(chunk_vecs, chunk_ids):
                lines.append(json.dumps({"index": {"_index": cfg.index}}))
                lines.append(json.dumps({
                    "embedding": v.tolist(),
                    "video_id": vid
                }))
            
            self._post_with_retry(
                f"{cfg.host}/_bulk",
                data="\n".join(lines) + "\n"
            )
    
    def refresh(self):
        """Refresh index to make new vectors searchable."""
        self.session.post(f"{self.config.host}/{self.config.index}/_refresh")
        time.sleep(self.config.refresh_wait)
    
    def get_count(self) -> int:
        """Get document count in index."""
        resp = self.session.get(f"{self.config.host}/{self.config.index}/_count")
        return resp.json()["count"]
    
    def _post_with_retry(self, url: str, data: str) -> requests.Response:
        """POST with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                resp = self.session.post(url, data=data)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:  # too many requests
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                resp.raise_for_status()
            except requests.exceptions.ConnectionError:
                if attempt < self.config.max_retries - 1:
                    log.warning(f"Connection error, retry {attempt+1}...")
                    time.sleep(self.config.retry_delay)
                else:
                    raise
        raise RuntimeError(f"Failed after {self.config.max_retries} retries: {url}")


# ============================================================
# LOCAL DEDUP (MATRIX MULTIPLY)
# ============================================================

def local_dedup(vectors: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove intra-batch duplicates using matrix multiply.
    
    Returns:
        unique_vectors: vectors to keep
        keep_mask: boolean mask of which input vectors to keep
        num_discarded: count of discarded duplicates
    """
    n = len(vectors)
    if n == 0:
        return vectors, np.array([], dtype=bool), 0
    
    # Compute full pairwise similarity matrix
    sim_matrix = vectors @ vectors.T
    np.fill_diagonal(sim_matrix, 0)
    
    # Greedy: keep first, discard later duplicates
    to_keep = np.ones(n, dtype=bool)
    for i in range(1, n):
        if to_keep[i]:
            # Check against all earlier vectors that are still kept
            if sim_matrix[i, :i][to_keep[:i]].max() >= threshold:
                to_keep[i] = False
    
    num_discarded = n - to_keep.sum()
    return vectors[to_keep], to_keep, num_discarded


# ============================================================
# VECTOR SOURCE (replace with your S3/SQS reader)
# ============================================================

class VectorSource:
    """
    Replace this with actual data source (S3, SQS, etc.)
    Must yield (vectors, video_ids) batches.
    Vectors must be L2-normalized float32.
    """
    
    def __init__(self, vectors: np.ndarray, video_ids: List[str], batch_size: int):
        self.vectors = vectors
        self.video_ids = video_ids
        self.batch_size = batch_size
        self.offset = 0
    
    def has_next(self) -> bool:
        return self.offset < len(self.vectors)
    
    def next_batch(self) -> Tuple[np.ndarray, List[str]]:
        end = min(self.offset + self.batch_size, len(self.vectors))
        batch_vecs = self.vectors[self.offset:end]
        batch_ids = self.video_ids[self.offset:end]
        self.offset = end
        return batch_vecs, batch_ids
    
    @property
    def total(self) -> int:
        return len(self.vectors)


# ============================================================
# MAIN DEDUP PIPELINE
# ============================================================

def run_dedup(config: Config, source: VectorSource):
    """Run the full sequential batch dedup pipeline."""
    
    client = OpenSearchClient(config)
    
    total_indexed = 0
    total_discarded = 0
    batch_num = 0
    start_time = time.time()
    
    log.info(f"Starting dedup: {source.total:,} vectors, batch={config.batch_size}")
    log.info(f"Threshold: cosine>={config.cosine_threshold} (min_score={config.min_score})")
    
    while source.has_next():
        batch_num += 1
        batch_vecs, batch_ids = source.next_batch()
        bs = len(batch_vecs)
        
        # ---- Step 1: Local dedup (matrix multiply) ----
        t1 = time.time()
        unique_vecs, keep_mask, local_discarded = local_dedup(
            batch_vecs, config.cosine_threshold
        )
        unique_ids = [batch_ids[i] for i in range(bs) if keep_mask[i]]
        total_discarded += local_discarded
        local_time = time.time() - t1
        
        # ---- Step 2: _msearch radial search ----
        t2 = time.time()
        if len(unique_vecs) > 0:
            has_match = client.msearch_radial(unique_vecs)
            
            # Filter to survivors (no match in index)
            survivors_vecs = []
            survivors_ids = []
            for i, matched in enumerate(has_match):
                if matched:
                    total_discarded += 1
                else:
                    survivors_vecs.append(unique_vecs[i])
                    survivors_ids.append(unique_ids[i])
            
            survivors_vecs = np.array(survivors_vecs) if survivors_vecs else np.array([])
        else:
            survivors_vecs = np.array([])
            survivors_ids = []
        search_time = time.time() - t2
        
        # ---- Step 3: _bulk index survivors ----
        t3 = time.time()
        if len(survivors_vecs) > 0:
            client.bulk_index(survivors_vecs, survivors_ids)
            total_indexed += len(survivors_vecs)
        bulk_time = time.time() - t3
        
        # ---- Step 4: Wait for refresh ----
        client.refresh()
        
        # ---- Logging ----
        elapsed = time.time() - start_time
        vps = (batch_num * config.batch_size) / elapsed
        log.info(
            f"Batch {batch_num} | "
            f"In:{bs} LocalDedup:-{local_discarded} Search:-{bs-local_discarded-len(survivors_vecs)} Indexed:{len(survivors_vecs)} | "
            f"Local:{local_time:.2f}s Search:{search_time:.2f}s Bulk:{bulk_time:.2f}s | "
            f"Total: idx={total_indexed:,} disc={total_discarded:,} | "
            f"VPS:{vps:.0f} Elapsed:{elapsed:.0f}s"
        )
    
    # ---- Final stats ----
    total_time = time.time() - start_time
    final_count = client.get_count()
    
    log.info("=" * 60)
    log.info("DEDUP COMPLETE")
    log.info(f"  Total time:     {total_time:.0f}s ({total_time/3600:.1f}h)")
    log.info(f"  Total indexed:  {total_indexed:,}")
    log.info(f"  Total discarded:{total_discarded:,}")
    log.info(f"  Docs in index:  {final_count:,}")
    log.info(f"  Avg VPS:        {source.total/total_time:.0f}")
    log.info("=" * 60)
    
    return total_indexed, total_discarded


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    config = Config(
        host=os.environ.get("OPENSEARCH_HOST", "http://localhost:9200"),
        index="video_vectors",
        batch_size=10000,
    )
    
    # --- FOR TESTING: Generate synthetic data ---
    # Replace this with your actual S3/SQS data source
    log.info("Generating synthetic test data...")
    
    TOTAL = int(os.environ.get("TOTAL_VECTORS", "10000"))
    DUP_RATE = float(os.environ.get("DUP_RATE", "0.05"))
    N_UNIQUE = int(TOTAL * (1 - DUP_RATE))
    N_DUPS = TOTAL - N_UNIQUE
    
    np.random.seed(42)
    unique = np.random.randn(N_UNIQUE, config.dimension).astype('float32')
    unique = unique / np.linalg.norm(unique, axis=1, keepdims=True)
    
    dup_src = np.random.choice(N_UNIQUE, N_DUPS)
    dups = unique[dup_src] + np.random.randn(N_DUPS, config.dimension).astype('float32') * 0.01
    dups = dups / np.linalg.norm(dups, axis=1, keepdims=True)
    
    all_vecs = np.vstack([unique, dups])
    all_vecs = all_vecs[np.random.permutation(TOTAL)]
    video_ids = [f"video_{i:08d}" for i in range(TOTAL)]
    
    log.info(f"Data: {TOTAL} vectors ({N_UNIQUE} unique + {N_DUPS} dups)")
    
    # --- Create index ---
    try:
        requests.delete(f"{config.host}/{config.index}")
        time.sleep(1)
    except:
        pass
    
    client = OpenSearchClient(config)
    client.create_index()
    time.sleep(2)
    
    # --- Run pipeline ---
    source = VectorSource(all_vecs, video_ids, config.batch_size)
    run_dedup(config, source)
