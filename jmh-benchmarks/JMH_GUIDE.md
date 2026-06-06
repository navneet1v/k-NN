# JMH Benchmark Guide

A practical reference for writing and interpreting JMH microbenchmarks in this module.

---

## Benchmark Modes

JMH supports multiple measurement modes via `@BenchmarkMode`. Each answers a different question.

| Mode | What it measures | Unit | When to use |
|------|-----------------|------|-------------|
| `Mode.Throughput` | Operations per time unit | ops/s, ops/ms, ops/us | "How many blocks can I score per second?" — most common for hot loops |
| `Mode.AverageTime` | Average time per operation | ns/op, us/op, ms/op | "How long does one scoreBlock call take on average?" |
| `Mode.SampleTime` | Time distribution (percentiles) | ns/op with p50/p99/p999 | "What's my tail latency?" — good for detecting GC pauses or JIT deopt |
| `Mode.SingleShotTime` | Time for a single invocation (no warmup) | ms/op | "How long does cold startup take?" — measures interpreter + C1 performance |
| `Mode.All` | Runs all modes | mixed | Exploratory — generates a lot of data |

### Choosing a Mode

- **Throughput** for comparing two implementations of the same operation (our primary use case)
- **AverageTime** when you want latency numbers directly (easier to reason about than ops/us)
- **SampleTime** when you suspect variance or GC impact
- **SingleShotTime** for one-off operations like index loading or segment merge

You can combine modes:
```java
@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
```

---

## Time Units

`@OutputTimeUnit` controls how results are displayed.

| Unit | Good for |
|------|----------|
| `TimeUnit.NANOSECONDS` | Sub-microsecond operations (single distance computation) |
| `TimeUnit.MICROSECONDS` | Block-level operations (scoring 32 vectors) |
| `TimeUnit.MILLISECONDS` | Segment-level operations (scanning a full posting list) |
| `TimeUnit.SECONDS` | Bulk scans (millions of vectors) |

Rule of thumb: pick the unit where your numbers are in the 1–1000 range.

---

## State and Scope

`@State` defines how benchmark state (fields) is shared across threads.

| Scope | Meaning | When to use |
|-------|---------|-------------|
| `Scope.Benchmark` | One instance shared by all threads | Default. Use for read-only data (vectors, pre-built indexes) |
| `Scope.Thread` | One instance per thread | When each thread needs its own mutable scratch buffers |
| `Scope.Group` | One instance per thread group | Producer-consumer benchmarks |

For our scoring benchmarks, `Scope.Benchmark` is correct — we're measuring single-threaded block scoring throughput.

---

## Setup and Teardown

| Level | When it runs | Use for |
|-------|-------------|---------|
| `Level.Trial` | Once per fork (before all iterations) | Expensive init: generate vectors, write quantized blocks, build directories |
| `Level.Iteration` | Before each measurement iteration | Resetting mutable state, re-randomizing data |
| `Level.Invocation` | Before each benchmark method call | **Avoid** — adds measurement overhead. Only for truly expensive setup (disk I/O per call) |

```java
@Setup(Level.Trial)
public void setup() { /* runs once per fork */ }

@TearDown(Level.Trial)
public void teardown() { /* cleanup after all iterations in this fork */ }
```

---

## Fork, Warmup, Measurement

### @Fork

```java
@Fork(value = 3, jvmArgsAppend = {"-Xmx2g", "--add-modules=jdk.incubator.vector"})
```

- **value** — Number of separate JVM processes. Each fork starts fresh (no JIT history, no heap state from prior runs). More forks = more confidence that results aren't an artifact of one JIT compilation path.
- **jvmArgsAppend** — Extra JVM flags for forked processes. Critical for enabling incubator modules.
- **jvmArgsPrepend** — Flags prepended (before JMH's own flags). Use for `-XX:` flags that must come early.

| Forks | Use case |
|-------|----------|
| 1 | Quick dev iteration — fast but noisy |
| 3 | Reasonable confidence for comparison |
| 5+ | Publishable results |

### @Warmup

```java
@Warmup(iterations = 4, time = 2)
```

- **iterations** — Number of warmup rounds (results discarded)
- **time** — Duration of each warmup round (seconds)

Warmup lets the JIT compile hot paths to C2, stabilize branch predictors, and fill caches. For vector scoring code, 3-4 iterations of 2s each is sufficient.

### @Measurement

```java
@Measurement(iterations = 5, time = 2)
```

- **iterations** — Number of measured rounds (results reported)
- **time** — Duration of each measurement round

More iterations reduce noise. For stable microbenchmarks (no GC, no I/O), 5 iterations is fine. For noisier benchmarks, use 10+.

---

## Parameters

`@Param` creates a benchmark matrix — JMH runs every combination.

```java
@Param({"128", "768", "1024"})
int dimension;

@Param({"1", "2", "4"})
byte docBits;
```

This generates 3 × 3 = 9 benchmark configurations, each getting its own warmup + measurement cycles.

**Tip:** For quick dev runs, override from the command line:
```bash
-p dimension=768 -p docBits=1
```

---

## Preventing Dead-Code Elimination

The JIT will eliminate code whose result is never used. Always:

1. **Return a value** from `@Benchmark` methods:
   ```java
   @Benchmark
   public float scoreBlock() { ... return collector.minCompetitiveSimilarity(); }
   ```

2. **Use `Blackhole.consume()`** for multiple results:
   ```java
   @Benchmark
   public void multiResult(Blackhole bh) {
       bh.consume(score1);
       bh.consume(score2);
   }
   ```

3. **Never store results in a field** without returning or consuming them.

---

## Writing a Simple Benchmark

```java
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 3, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class MyBenchmark {

    @Param({"128", "768"})
    int dimension;

    private float[] a;
    private float[] b;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42L);
        a = new float[dimension];
        b = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            a[i] = rng.nextFloat();
            b[i] = rng.nextFloat();
        }
    }

    @Benchmark
    public float dotProduct() {
        float sum = 0f;
        for (int i = 0; i < dimension; i++) sum += a[i] * b[i];
        return sum;
    }

    @Benchmark
    public float dotProductVectorUtil() {
        return VectorUtil.dotProduct(a, b);
    }
}
```

---

## Reading JMH Output

```
Benchmark                              (dimension)  Mode  Cnt    Score    Error  Units
MyBenchmark.dotProduct                         768  thrpt   15   12.345 ±  0.123  ops/us
MyBenchmark.dotProductVectorUtil               768  thrpt   15   48.678 ±  0.456  ops/us
```

| Column | Meaning |
|--------|---------|
| Benchmark | Fully qualified method name |
| (dimension) | Parameter value for this row |
| Mode | thrpt=throughput, avgt=average time, sample=percentiles |
| Cnt | Total measurement iterations across all forks (forks × iterations) |
| Score | The measured value (higher is better for throughput, lower for time) |
| Error | ± margin at 99.9% confidence interval |
| Units | ops/us, ns/op, etc. |

**Interpreting results:**
- If the error bars overlap between two benchmarks, the difference is **not statistically significant**
- A 2x throughput difference with tight error bars is a real signal
- A 5% difference with wide error bars needs more forks

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Measuring JIT compilation | First fork is 10x slower | Use ≥3 forks, adequate warmup |
| Dead-code elimination | Unrealistically high throughput | Return a value or use Blackhole |
| Constant folding | JIT precomputes the answer | Use `@State` fields, not local constants |
| Loop optimization | JIT unrolls and eliminates the loop | Use `@Param` for sizes, not hardcoded constants |
| Allocating in @Benchmark | GC noise in measurements | Move allocation to `@Setup` |
| Level.Invocation setup | Measures setup time too | Use Level.Trial or Level.Iteration |

---

## Useful Command-Line Flags

```bash
# List all benchmarks
-l

# Run only benchmarks matching regex
ScoreBlock

# Override parameters
-p dimension=768 -p docBits=1

# Control forks/warmup/measurement
-f 5 -wi 4 -i 10

# JSON output for programmatic comparison
-rf json -rff results.json

# Thread count
-t 1

# Profilers (GC, stack, perf)
-prof gc                    # GC allocation rate
-prof stack                 # Stack sampling (hottest methods)
-prof perfnorm              # Linux perf (cycles, cache misses per op)
-prof jfr                   # Java Flight Recorder

# Dry run (compile check, no actual measurement)
-f 0 -wi 0 -i 1
```

---

## Profiling Tips

For understanding *why* one method is faster:

```bash
# GC allocation pressure
./gradlew :jmh-benchmarks:jmh -Pjmh.args="ScoreBlock -f 1 -prof gc -p dimension=768 -p docBits=1"

# Hottest methods (is JIT vectorizing?)
./gradlew :jmh-benchmarks:jmh -Pjmh.args="ScoreBlock -f 1 -prof stack -p dimension=768"

# Print JIT compilation log (look for vectorize/auto-vectorization messages)
# Add to @Fork jvmArgsAppend: "-XX:+PrintCompilation", "-XX:+UnlockDiagnosticVMOptions", "-XX:+PrintInlining"
```
