# AGENTS.md — jmh-benchmarks module

Instructions for AI agents working in this module.

## Purpose

This is a standalone JMH microbenchmark module for the OpenSearch k-NN plugin. It benchmarks low-level scoring functions in isolation (no cluster, no indexing, no network).

## Module Structure

```
jmh-benchmarks/
├── build.gradle                          # Gradle config (java-library + JMH deps)
├── README.md                             # Human-readable usage guide
├── AGENTS.md                             # This file
└── src/main/java/org/opensearch/knn/benchmark/
    ├── Main.java                         # JMH entry point (delegates to org.openjdk.jmh.Main)
    └── ScoreBlockBenchmark.java          # Block scoring comparison benchmark
```

## Build & Run Commands

```bash
# Compile only (no tests, no fat JAR):
./gradlew :jmh-benchmarks:compileJava

# Full assemble (produces fat JAR):
./gradlew :jmh-benchmarks:assemble

# Run benchmarks via Gradle:
./gradlew :jmh-benchmarks:jmh -Pjmh.args="<regex> <jmh-flags>"

# Run via fat JAR (after assemble):
java --add-modules=jdk.incubator.vector -jar jmh-benchmarks/build/libs/jmh-benchmarks-*-jmh.jar <regex> <jmh-flags>
```

## This Module Is NOT Part of the Main Build

`./gradlew build`, `./gradlew assemble`, `./gradlew test` on the root project should NOT trigger this module. If it does, that's a bug — fix it by ensuring root lifecycle tasks don't depend on `:jmh-benchmarks`.

## Key Conventions

1. **One benchmark class per concern.** Don't mix unrelated benchmarks in a single file.
2. **Use `@Param` for dimensions, bit widths, and metrics.** This produces a full matrix automatically.
3. **`@Setup(Level.Trial)` for expensive init** (directory creation, vector generation). Never allocate in the `@Benchmark` method.
4. **Return a value from `@Benchmark` methods** to prevent JIT dead-code elimination. Returning `collector.minCompetitiveSimilarity()` or a float score is sufficient.
5. **Use `ByteBuffersDirectory`** (in-memory) for block-level benchmarks. We're measuring CPU scoring, not disk I/O.
6. **Always include `--add-modules=jdk.incubator.vector`** in `@Fork(jvmArgsAppend=...)` so Panama Vector API is available.

## Dependencies

- `project(':')` — the root k-NN plugin (all ClusterANN codec classes)
- `org.openjdk.jmh:jmh-core:1.37`
- `org.openjdk.jmh:jmh-generator-annprocess:1.37` (annotation processor)

## Adding a New Benchmark

1. Create `src/main/java/org/opensearch/knn/benchmark/YourBenchmark.java`
2. Annotate the class:
   ```java
   @BenchmarkMode(Mode.Throughput)
   @OutputTimeUnit(TimeUnit.MICROSECONDS)
   @State(Scope.Benchmark)
   @Warmup(iterations = 4, time = 2)
   @Measurement(iterations = 5, time = 2)
   @Fork(value = 3, jvmArgsAppend = {"-Xmx2g", "-Xms2g", "--add-modules=jdk.incubator.vector"})
   ```
3. Add `@Param` fields for the parameter matrix.
4. Add `@Setup(Level.Trial)` method to generate data.
5. Add `@Benchmark` methods that return a value.

## Test-Friendly Classes

`ClusterANNFieldState` has a public constructor `ClusterANNFieldState(int dimension, byte docBits)` for creating minimal instances in benchmarks/tests without reading `.clam` files.

## What NOT to Do

- Don't add integration tests here. This is strictly for JMH microbenchmarks.
- Don't add dependencies on OpenSearch test framework or any test libraries.
- Don't create network/cluster fixtures. If you need segment-level benchmarks, write blocks to a `ByteBuffersDirectory`.
- Don't modify `Main.java` — it's a one-liner delegate to `org.openjdk.jmh.Main`.
