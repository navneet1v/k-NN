#!/usr/bin/env python3
"""Sequential Batch Dedup on OpenSearch - 10K vectors, innerproduct"""
import numpy as np, requests, json, time, sys

HOST = 'http://localhost:9200'
INDEX = 'video_dedup'
DIM = 768
TOTAL = 10000
N_UNIQUE = 7000
N_DUPS = 3000
COSINE_THRESHOLD = 0.95
# OpenSearch faiss IP score = 1 + dot_product (for normalized vectors)
# cosine >= 0.95 means score >= 1.95
MIN_SCORE = 1.0 + COSINE_THRESHOLD  # = 1.95
BATCH_SIZE = 100

print(f'OpenSearch min_score for cosine>={COSINE_THRESHOLD}: {MIN_SCORE}', flush=True)
print('Generating data...', flush=True)
np.random.seed(42)
unique = np.random.randn(N_UNIQUE, DIM).astype('float32')
unique = unique / np.linalg.norm(unique, axis=1, keepdims=True)
dup_src = np.random.choice(N_UNIQUE, N_DUPS)
dups = unique[dup_src] + np.random.randn(N_DUPS, DIM).astype('float32') * 0.005
dups = dups / np.linalg.norm(dups, axis=1, keepdims=True)
all_vecs = np.vstack([unique, dups])[np.random.permutation(TOTAL)]
print(f'Data: {TOTAL} vecs ({N_UNIQUE} unique + {N_DUPS} dups)', flush=True)

requests.delete(f'{HOST}/{INDEX}')
time.sleep(1)
resp = requests.put(f'{HOST}/{INDEX}', json={
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0, 'refresh_interval': '1s', 'knn': True}},
    'mappings': {'properties': {'embedding': {
        'type': 'knn_vector', 'dimension': DIM, 'data_type': 'float',
        'method': {'name': 'hnsw', 'engine': 'faiss', 'space_type': 'innerproduct',
                   'parameters': {'m': 16, 'ef_construction': 128}}}}}
})
print(f'Index created: {resp.status_code}', flush=True)
time.sleep(2)

indexed = 0; discarded = 0; t0 = time.time()
for bn, s in enumerate(range(0, TOTAL, BATCH_SIZE)):
    batch = all_vecs[s:s+BATCH_SIZE]
    lines = []
    for v in batch:
        lines.append(json.dumps({'index': INDEX}))
        lines.append(json.dumps({'size':1,'min_score':MIN_SCORE,
                                 'query':{'knn':{'embedding':{'vector':v.tolist(),'k':1}}}}))
    r = requests.post(f'{HOST}/_msearch', data='\n'.join(lines)+'\n',
                      headers={'Content-Type':'application/x-ndjson'})
    if r.status_code != 200:
        print(f'FAIL batch {bn}: {r.status_code} {r.text[:200]}', flush=True); break

    results = r.json()['responses']
    cands = [batch[i] for i,res in enumerate(results) if res.get('hits',{}).get('total',{}).get('value',0)==0]
    discarded += (len(batch) - len(cands))

    survs = []; buf = []
    for v in cands:
        if buf and (np.array(buf)@v).max() >= COSINE_THRESHOLD:
            discarded += 1
        else:
            buf.append(v); survs.append(v)

    if survs:
        bl = []
        for v in survs:
            bl.append(json.dumps({'index':{'_index':INDEX}}))
            bl.append(json.dumps({'embedding':v.tolist()}))
            indexed += 1
        requests.post(f'{HOST}/_bulk', data='\n'.join(bl)+'\n',
                      headers={'Content-Type':'application/x-ndjson'})
    time.sleep(1)
    if (bn+1) % 20 == 0 or bn == 0:
        print(f'Batch {bn+1:3d}/100 | Idx:{indexed:5d} Disc:{discarded:5d} | {time.time()-t0:.0f}s', flush=True)

print(f'\nRESULT: Indexed={indexed}, Discarded={discarded}, Time={time.time()-t0:.0f}s', flush=True)
print(f'Expected: ~7000 indexed, ~3000 discarded', flush=True)

# Verify
print('\nVerifying...', flush=True)
requests.post(f'{HOST}/{INDEX}/_refresh')
time.sleep(1)
count = requests.get(f'{HOST}/{INDEX}/_count').json()['count']
print(f'Docs in index: {count}', flush=True)

# Sample check
sample = requests.post(f'{HOST}/{INDEX}/_search', json={'size':100,'query':{'match_all':{}}}).json()
leaked = 0
for hit in sample['hits']['hits'][:50]:
    vec = hit['_source']['embedding']
    r = requests.post(f'{HOST}/{INDEX}/_search', json={
        'size':2,'query':{'knn':{'embedding':{'vector':vec,'k':2}}}}).json()
    hits = r['hits']['hits']
    if len(hits)>=2 and hits[1]['_score'] >= MIN_SCORE:
        leaked += 1
print(f'Leaked duplicates in sample of 50: {leaked}', flush=True)
print('PASS!' if leaked==0 else f'FAIL: {leaked} leaked', flush=True)
