# k-NN JMH Benchmarks

Microbenchmarks for the OpenSearch k-NN plugin using [JMH](https://openjdk.org/projects/code-tools/jmh/).

This module is **not** built as part of the main plugin build. It must be explicitly targeted.

## Build

```bash
./gradlew :jmh-benchmarks:assemble
```

This produces a fat JAR at `jmh-benchmarks/build/libs/jmh-benchmarks-*-jmh.jar`.

## Run

### Via Gradle (development)

```bash
# Run all benchmarks with defaults
./gradlew :jmh-benchmarks:jmh

# Run specific benchmark with custom args
./gradlew :jmh-benchmarks:jmh -Pjmh.args="ScoreBlock -f 1 -wi 2 -i 3"

# Run with specific parameters
./gradlew :jmh-benchmarks:jmh -Pjmh.args="ScoreBlock -f 3 -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

### Via JAR (production benchmarking)

```bash
java --add-modules=jdk.incubator.vector \
     -Djava.library.path=jni/build/release \
     -jar jmh-benchmarks/build/libs/jmh-benchmarks-*-jmh.jar
```

## JMH Options

| Flag | Description | Example |
|------|-------------|---------|
| `-l` | List available benchmarks | |
| `-f N` | Number of forks (separate JVM processes) | `-f 3` |
| `-wi N` | Warmup iterations per fork | `-wi 4` |
| `-i N` | Measurement iterations per fork | `-i 5` |
| `-t N` | Threads | `-t 1` |
| `-p key=v1,v2` | Parameterize | `-p dimension=128,768` |
| `-rf json` | Output format | `-rf json -rff results.json` |
| `regexp` | Run benchmarks matching regex | `ScoreBlock` |

## Profiling

JMH supports several profilers to understand where time is spent.

### Stack sampling (quick, no external deps)

```bash
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 1 -prof stack -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

Prints the hottest methods inline. Fast but imprecise for short methods.

### JFR (Java Flight Recorder)

```bash
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 1 -prof jfr -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

Produces a `.jfr` file per fork. Open with `jmc` (Java Mission Control) or IntelliJ's profiler.

### async-profiler (most accurate for SIMD/native code)

```bash
# macOS (Homebrew install)
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 1 -prof async:libPath=/opt/homebrew/opt/async-profiler/lib/libasyncProfiler.dylib;output=flamegraph;dir=profile-results -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"

# Linux
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 1 -prof async:libPath=/path/to/libasyncProfiler.so;output=flamegraph;dir=profile-results -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

Produces an HTML flamegraph in `profile-results/`. Most accurate for hot loops with native/SIMD code.

### GC allocation pressure

```bash
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 1 -prof gc -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

Reports allocation rate (bytes/op) — useful to confirm zero-alloc in the hot path.

### Percentile latency (SampleTime mode)

```bash
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 3 -bm sample -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

Reports p50/p90/p99/p999 per invocation — reveals tail latency differences hidden by throughput mode.

## Available Benchmarks

### ScoreBlockBenchmark

Benchmarks ADC block scoring by scanning 20 consecutive blocks (640 vectors) under a single centroid — isolating the pure scoring cost (dot product + corrections + collection) with query quantization pre-computed in setup.

Two variants:
- `scoreBlock` — baseline: `readBytes` into heap `byte[]`, single-vector `VarHandle`-based dot product
- `scoreBlock_bulkSIMD` — optimized: zero-copy `MemorySegment` + Panama Vector API bulk4 dot product

**Parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `dimension` | 768, 1024 | Vector dimensionality |
| `docBits` | 1 | Scalar quantization bits (32x compression) |
| `metric` | EUCLIDEAN, MAXIMUM_INNER_PRODUCT, DOT_PRODUCT | Similarity function |
| `numBlocks` | 20 | Blocks per invocation (override: `-p numBlocks=100`) |

**Results (768-dim, 1-bit, MIP, Apple Silicon ARM NEON 128-bit):**

| Path | ops/ms | Speedup |
|------|--------|---------|
| `scoreBlock` (baseline) | 18.4 | 1.0x |
| `scoreBlock_bulkSIMD` | 73.9 | **4.0x** |

**Example: benchmark only the 768-dim 1-bit IP case (most common production config)**

```bash
./gradlew :jmh-benchmarks:jmh \
  -Pjmh.args="ScoreBlock -f 5 -wi 5 -i 10 -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
```

## Adding New Benchmarks

1. Create a new class in `src/main/java/org/opensearch/knn/benchmark/`
2. Annotate with JMH annotations (`@Benchmark`, `@State`, `@Param`, etc.)
3. Use `@Fork(jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})` if Panama Vector API is needed
4. Return a value from `@Benchmark` methods to prevent dead-code elimination

## Dependencies

This module depends on `:` (the root k-NN plugin) for access to all ClusterANN codec classes. JMH core and annotation processor are the only additional dependencies.

## Notes

- The JNI native library (`libKNNIndexV2_0_6`) must be built before running benchmarks that exercise native SIMD paths. Run `./gradlew buildJniLib` first if needed.
- For stable results, close other applications, disable turbo boost, and pin CPU frequency.
- Use `-f 5` or more forks for publishable numbers. Single-fork runs are fine for A/B iteration during development.
