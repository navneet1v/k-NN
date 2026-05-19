# ClusterANN Benchmarks

## Datasets (at /Users/viktari/pysptag/data/)
| Dataset | Vectors | Dims | Metric | Notes |
|---------|---------|------|--------|-------|
| sift | 1M | 128 | L2 | Standard ANN benchmark |
| cohere | 1M | 768 | IP | Unnormalized embeddings |
| gist | 1M | 960 | L2 | High-dimensional |

## Scripts
- `bench_all.py` — Run all datasets with configurable oversample/compression
- `bench_clusterann_sift.py` — SIFT-1M standalone
- `bench_clusterann_cohere.py` — Cohere-1M standalone

## Results
See `results/` folder for saved outputs.
