# HNSW Locality Reordering — Full Context Handoff

Written 2026-09-12. Everything needed to continue the work.

---

## 1. Objective / current state

Feature: HNSW locality-aware vector reordering for AOSS V2 (Juno), backported from
GitHub `navneet1v/k-NN@hnsw_locality`. A/B benchmark baseline (stock codec) vs candidate
(reordered codec) on gamma. Custom-deploy path (KNN-jar-only) is working.

**Two open workstreams right now:**
1. **PageTouchTracker enablement** — trying to get per-query read-amplification lines into
   worker logs to quantify the reordering benefit. NOT YET WORKING (no PageTouch lines appear).
2. **10M shuffled dataset indexing** — upload `documents-10m-shuffled.hdf5` to S3, download to
   both clients, index into a NEW index `target_index_10_shuffled` on both collections. Only the
   S3 upload was started (status uncertain).

---

## 2. Accounts / regions / roles

- Collection account: **199552501713** (us-east-1, stage gamma). Role: `Admin` via
  `ada credentials update --account 199552501713 --provider isengard --role Admin --once`.
- DP account (for search-deploy / fm-deployment): **326559903869**. Role Admin via
  `ada credentials update --account=326559903869 --role=Admin --once`.
- Beholder logs account: **372537475017**. Use role **ReadOnly** (Admin federation denied):
  `ada credentials update --account 372537475017 --provider isengard --role ReadOnly --once`.

## 3. Collections (NextGen V2, 32/32 OCU, standby ENABLED, GPU/remote-index-build disabled at index level)

| role | name | collection id | placement constraint (PC) | vector field | codec |
|------|------|---------------|---------------------------|--------------|-------|
| baseline  | locality-test-baseline  | tyc9hkty2lpuzdmlh7r4 | `kVjcRcJ_QK8C0BSK3YJ0pAcP9b91AzKM1LUL3_3xnPg` | `target_field` | stock Faiss1030ScalarQuantized |
| candidate | locality-test-candidate | eg30c89wykac88a5dbx5 | `6QARFhZ5vkNcbLoCaZgmdOiglvEl49PmqXsKjwjgZvo` | `reordering_target_field` | Faiss1030SQHNSWReordered (reordering gated on field name containing "reordering") |

- Existing indexes on BOTH: `target_index` (1M), `target_index_10m` (10M, 8 shards).
- Endpoints (for awscurl on clients): `https://<collection-id>.us-east-1.aoss.amazonaws.com`
  (baseline `tyc9hkty2lpuzdmlh7r4...`, candidate `eg30c89wykac88a5dbx5...`).
- NOTE: client role got **403 on `_mapping` / `_settings`** (data-access policy grants data ops,
  not those admin GETs). `_search` and bulk index DO work.

## 4. Benchmark clients (SSM-managed, NO SSH)

- baseline client EC2: `i-00b80d379fcad1466`
- candidate client EC2: `i-03f81c89c7522111a`
- c5.4xlarge, AMI ami-0ab5ebcd9152232a9, OSB 2.2.0.
- IAM role on clients: `locality-bench-client-role` (SSM + S3 + aoss:APIAccessAll); it is in both
  collections' data-access policies.
- pyenv on box: `export PYENV_ROOT=/home/ec2-user/.pyenv; export PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH`
- Signed AOSS calls via `awscurl --service aoss --region us-east-1 ...`
- Query vectors on box: `/home/ec2-user/qvec.json` (1 vector), `/home/ec2-user/qvecs100.json` (100).
- SSM gotcha: to run a multi-line script, base64-encode it and pass as ONE arg:
  `B64=$(base64 -w0 script.sh)` then
  `aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript --parameters "{\"commands\":[\"echo '$B64' | base64 -d > /tmp/s.sh\",\"chmod +x /tmp/s.sh\",\"su - ec2-user -c '/tmp/s.sh ARGS'\"]}"`
  (Do NOT `echo $(cat b64)` — it splits tokens and executes the base64 as commands. That already burned time.)

## 5. Workspaces / branches

- KNN-search workspace (make code changes + build jar here):
  `/workplace/workspace/reordering/KNN-search/src/SearchServicesESKNN`
  - Build FROM PACKAGE ROOT: `cd .../SearchServicesESKNN && brazil-build release`
    (needs `gradle_cache.zip` at package root; running from `knn/` fails).
  - `:test` task flakes (`FileNotFoundException output.bin.idx`) — compile+jar still succeed.
  - Output jar: `knn/build/plugins/knn/opensearch-knn-3.3.0.0.jar`
  - Remote branch `origin/hnsw_locality-search` (local branch `hnsw_locality`).
  - Repo: `ssh://git.amazon.com/pkg/SearchServicesESKNN`. Also branch `hnsw_locality-indexing`.
- KNN-indexing workspace: `/workplace/workspace/reordering/KNN-indexing/src/SearchServicesESKNN`
- Custom-deploy workspace (run search-deploy from here):
  `/workplace/workspace/reordering/customdeploy-search/src/JunoSearchWorkerDistribution`
  - JunoSearchWorker package REMOVED from this workspace (only JunoSearchWorkerDistribution) to
    avoid a `[JDK25]` resolution error during the dist build.

## 6. Custom deploy (KNN-jar-only) — WORKING

Deploy tool flags (added locally to `aoss-dp-dev-tools`, see runbook
`/home/navneev/knn-jar-replacement-aoss.md`): `--knn-jar-only` (`-koj`), `--knn-jar-path` (`-kojp`).
Overlays ONLY `opensearch-knn-*.jar` into image `plugins/knn/`. Pure-Java changes only (the C++
`.so` libs are NOT replaced).

Local tool edits (LOST on `toolbox update`; land a CR to `AOSSDPDevTools` to make durable):
`~/.toolbox/tools/aoss-dp-dev-tools/1.0.31249.0/lib/python3.10/site-packages/AOSSDPDevTools/aoss_dp_dev_tools/`
- `commons/input.py` (add_knn_jar_only, add_knn_jar_path)
- `deploy_tool/args.py` (fields + wiring)
- `deploy_tool/patches.py` (ESKNNPatch.apply_knn_jar_only)

### Deploy commands (RUN SEQUENTIALLY — never both at once)
Both share the same JunoSearchWorkerDistribution build dir; running both simultaneously makes one
deploy's `brazil-build clean` (`rm -rf build/`) delete the other's files mid-copy →
`cp: cannot stat .../lib/*.jar` → BUILD FAILED. That is exactly why a baseline deploy failed earlier.

```bash
ada credentials update --account=326559903869 --role=Admin --once
JAR=/workplace/workspace/reordering/KNN-search/src/SearchServicesESKNN/knn/build/plugins/knn/opensearch-knn-3.3.0.0.jar
cd /workplace/workspace/reordering/customdeploy-search/src/JunoSearchWorkerDistribution

# baseline (PC kVjcR...)
search-deploy -W -s gamma -r us-east-1 -dpc FMCell-00-DPCell-00 -R Admin \
  -wid default:199552501713:kVjcRcJ_QK8C0BSK3YJ0pAcP9b91AzKM1LUL3_3xnPg:SEARCH -v 3.3.0 \
  --knn-jar-only --knn-jar-path $JAR

# AFTER baseline prints "Started deployment", THEN candidate (PC 6QARF...)
search-deploy -W -s gamma -r us-east-1 -dpc FMCell-00-DPCell-00 -R Admin \
  -wid default:199552501713:6QARFhZ5vkNcbLoCaZgmdOiglvEl49PmqXsKjwjgZvo:SEARCH -v 3.3.0 \
  --knn-jar-only --knn-jar-path $JAR
```

### Track rollout
```bash
# FM deployment state (DP creds)
fm-deployment --region us-east-1 --stage gamma --data-plane-cell-id FMCell-00-DPCell-00 \
  describe --deployment-id <ID>          # DEPLOYING -> CURRENT
# worker software version (collection creds)
search-capacity --region us-east-1 --stage gamma -a 199552501713 -W -p <PC>
```

### Last successful deploys (with the DEBUG->INFO PageTouch jar)
- baseline deployment `5a1bcec2-21a7-4e5a-8a5e-c0d99ee96f15` -> CURRENT,
  software `upgradeCustomB6512044763_searchdev_navneev_1789235573`
- candidate deployment `30792570-a9ac-4b15-926c-fd0b82ce377d` -> CURRENT,
  software `upgradeCustomB6512044763_searchdev_navneev_1789235024`
Both workers 2x48 GCU, STEADY_STATE.

## 7. Uncommitted code changes in KNN-search working tree (as of handoff)

`git -C /workplace/workspace/reordering/KNN-search/src/SearchServicesESKNN status --short` shows:
- `M knn/.../index/codec/scorer/PageTouchTracker.java`
- `M knn/.../plugin/KNNPlugin.java`

### Change A — PageTouchTracker gate DEBUG -> INFO (DONE, built, deployed once)
File `knn/src/main/java/org/opensearch/knn/index/codec/scorer/PageTouchTracker.java`, method
`isEnabled()` now:
```java
public static boolean isEnabled() {
    return ENABLED_BY_PROP || log.isInfoEnabled();
}
```
(was `log.isDebugEnabled()`). Goal: always-on, no cluster-setting / JVM flag. Verified compiled
into jar (javap shows `isInfoEnabled`).

### Change B — KNNPlugin marker (DONE in source, NOT yet built/deployed at handoff)
File `knn/src/main/java/org/opensearch/knn/plugin/KNNPlugin.java` line 288:
```java
log.info("jar replacement with Pagetoucher change");
```
(was `log.info("coming from knn jar replacement");`). Purpose: prove THIS jar loads.
USER wants to run `brazil-build release` + the two deploys THEMSELVES. Do not build/deploy for them.

Also present on search branch (committed): per-vector prefetch `prefetchBasedOnOrdinals`
(commit acd53f3, backport of upstream 95a66004).

---

## 8. PageTouchTracker investigation — WHERE IT'S STUCK

Goal: get `PageTouch query#[...] readAmplification=[...]` INFO lines per query into
`schwrkr-ServerLogs-us-east-1-gamma`, for both baseline and candidate, to compare read amplification.

### Confirmed facts
- Jar loads: INFO marker `coming from knn jar replacement` appears from new deployment (o.o.k.p.KNNPlugin, level INFO).
- INFO is enabled on workers: billions of INFO `o.o.k` lines; only 13k WARN, 945 ERROR.
- `PageTouchTracker.end()` logs at INFO; exact template string is present in the deployed jar:
  `PageTouch query#[{}] context=[{}] dataUsedBytes=[{}] ...`
- `begin()/end()` bracket IS compiled into `FaissMemoryOptimizedSearcher` (javap confirms
  `PageTouchTracker.begin`/`end` invocations). Call chain in source:
  public `search(float[]...)` -> `getFloatValues()` -> private `search(enc,scorer,...)` which does
  `begin(); try { HnswGraphSearcher.search(...) } finally { end("nvec=...") }`.
- `PrefetchHelper.prefetch()` and `prefetchBasedOnOrdinals()` call
  `PageTouchTracker.get().recordBatch(...)` UNCONDITIONALLY before the prefetch-enabled check.

### Symptom
- Fired k=10 (5x) and k=100 (8x) queries on BOTH collections via awscurl — all returned OK.
- **ZERO `PageTouch` lines** appear in logs (searched last 7min, 90min — nothing).
- Observed instead: `o.o.k.i.c.s.PrefetchHelper` INFO `KNNPrefetchDBG SKIPPED (prefetch disabled): numOrds=NN`
  — BUT that string exists in NEITHER our source NOR our built jar. It comes from OTHER workers in
  the shared cell (c02, 26+ host streams) running a DIFFERENT knn build. i.e. the shared log group
  mixes many workers.
- Observed `o.o.k.m.MemorySegmentAddressExtractorUtil` WARN: "Failed to load
  MemorySegmentAddressExtractorJDK22, falling back to ...JDK21" (ClassNotFound, non-fatal). Stack:
  `FaissMemoryOptimizedSearcher.search(:82)` -> `FaissIdMapIndex.getFloatValues(:89)` ->
  `FaissHNSWIndex.getFloatValues(:62)` -> `FaissIndexScalarQuantizedFlat.getFloatValues(:105)` ->
  `MemorySegmentAddressExtractorUtil.<clinit>(:60)`. So the search uses the memory-optimized
  (mmap) path, NOT PrefetchHelper — meaning `recordBatch` likely never runs on these queries.

### Leading hypotheses (unverified — for next agent)
1. `end()` never runs because the queries never reach the private `search(...)` bracket on OUR
   workers — but logs show search works, so more likely:
2. Our two workers' log lines are hard to isolate in the shared `schwrkr-ServerLogs` group; the
   PageTouch line may be there but filtered out. **Next step: filter the Logs Insights query by the
   specific worker host / stream, not by PC or free text.** Get our workers' host IPs from
   `search-capacity ... -p <PC>` or from `fm-...`, then filter `@logStream` to those hosts.
3. The memory-optimized searcher reads vectors via MemorySegmentAddressExtractor and does NOT route
   through PrefetchHelper, so `recordBatch` isn't fed — but `end()` should STILL emit a (near-zero)
   line if `active==true`. If NO line at all, `begin()` set `active=false`, i.e. `isEnabled()` false
   at query time -> the PageTouchTracker class logger specifically is above INFO. **Next step:
   verify the effective level of logger `org.opensearch.knn.index.codec.scorer.PageTouchTracker`
   specifically (per-class log4j config may differ from package).** If so, either lower that logger
   or change `isEnabled()` to not depend on logger level at all (e.g. always true, or a static flag
   flipped on).
4. Simplest robust fix to consider: make PageTouch fire regardless of logger config AND ensure the
   feed path runs for the memory-optimized searcher (add a `recordBatch`/`recordOrd` call on the
   actual mmap read path in the memory-optimized scorer, not only PrefetchHelper).

### Useful Logs Insights recipe (Beholder ReadOnly, group schwrkr-ServerLogs-us-east-1-gamma)
```bash
ada credentials update --account 372537475017 --provider isengard --role ReadOnly --once
LG=schwrkr-ServerLogs-us-east-1-gamma; S=$(( $(date -u +%s) - 900 )); E=$(date -u +%s)
id=$(aws logs start-query --region us-east-1 --log-group-name "$LG" --start-time $S --end-time $E \
  --query-string 'fields @logStream,@message | filter @message like /PageTouch/ | limit 50' \
  --query queryId --output text)
# poll: aws logs get-query-results --region us-east-1 --query-id "$id" ... until status=Complete
```
JMESPath gotcha: quoting `@message` in `--query` fails; instead pull JSON and parse in python:
`... --output json | python3 -c 'import sys,json;[print(" | ".join(f["value"] for f in r if f["field"]!="@ptr")) for r in json.load(sys.stdin)["results"]]'`

---

## 9. 10M shuffled dataset task — STATUS

Requested: upload `/workplace/workspace/benchmark-cluster-ann/documents-10m-shuffled.hdf5`
(30.7 GB; HDF5 keys: `train` 10,000,000x768 float32, `test` 10000x768, `neighbors` 10000x100 int64)
to S3 bucket `knn-all-datasets` (bucket is in account **199552501713**, us-east-1), download onto
both clients, then index into NEW index **`target_index_10_shuffled`** on BOTH collections (same
mappings as `target_index_10m`, just different index name; baseline field `target_field`,
candidate field `reordering_target_field`).

- S3 UPLOAD: started as background `aws s3 cp` (was PID 29520, log `/tmp/upload_ds.log`,
  chunksize 256MB, 20 concurrent). **STATUS UNCERTAIN — verify with:**
  `aws s3 ls s3://knn-all-datasets/documents-10m-shuffled.hdf5 --region us-east-1` and
  `aws s3api list-multipart-uploads --bucket knn-all-datasets --region us-east-1`.
- Client DOWNLOAD: not started.
- Index create + bulk index: not started. Need the exact `target_index_10m` mapping/settings; client
  role 403s on `_mapping` GET, so either grant that action or reconstruct the create body (768D,
  8 shards, FAISS SQ HNSW; candidate field name must contain "reordering" to trigger reordered codec).

---

## 10. Artifacts / files

- Runbook (KNN-jar-only custom deploy + how to update the deploy tool):
  `/home/navneev/knn-jar-replacement-aoss.md`
- Benchmark results so far: `/workplace/workspace/reordering/hnsw-locality-benchmark-results.md`
  (Run 1 = 1M, Run 2 = 10M, Run 3 = prefetch; recall parity ~0.94; ~14% fewer block cache
  misses/loads for candidate; latency win not clean at 32-OCU because cache-resident).
- isenlink helper (shorten CW console URLs; needs internal CA bundle
  `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem` + midway cookie):
  `/home/navneev/.aim/packages/AOSStorageAgents-1.0/eventId-6513335236/context/scripts/isenlink.py`

## 11. Log group cheat-sheet (Beholder acct 372537475017, ReadOnly)
- `schwrkr-ServerLogs-us-east-1-gamma`   — search worker app/server logs (mixed across all workers in cell)
- `schwrkr-ServiceMetrics-us-east-1-gamma`
- `idxwrkr-ApplicationLogs-us-east-1-gamma` — indexing worker app logs
