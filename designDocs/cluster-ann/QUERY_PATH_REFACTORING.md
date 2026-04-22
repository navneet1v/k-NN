# Query Path Refactoring for Engine-Less Algorithms

**Status:** Future work

## Problem

The query path is tightly coupled to `KNNEngine` — it assumes every field has an engine with a library, native files, and engine-specific validation. Engine-less algorithms hit NPEs, early returns, and unsupported operation errors at multiple points:

1. `KNNQueryBuilder.doToQuery()` — NPE on `knnLibrary.getKNNLibrarySearchContext()` (currently guarded with `knnEngine != UNDEFINED`)
2. `KNNWeight.approximateSearch()` — Early return when no native engine files found (currently guarded with `EngineLessMethod.isEngineLess()`)
3. `MemoryOptimizedKNNWeight` — Hardcoded `KnnSearchStrategy.Hnsw(60)` and ACORN filtering thresholds
4. Radial search — "Engine [UNDEFINED] does not support radial search"

As we add more engine-less algorithms, each one would need ad-hoc guards scattered across the codebase.

## Design Principle

**Prefer composition over inheritance.** Instead of a single god interface or an inheritance hierarchy, break concerns into focused, composable pieces that algorithms assemble.

## Proposed Interfaces

```java
// Declares what the algorithm can and cannot do
public interface SearchCapabilities {
    boolean requiresNativeEngineFiles();
    boolean supportsRadialSearch();
}

// Provides the KnnSearchStrategy for collector creation
public interface SearchStrategyProvider {
    KnnSearchStrategy getSearchStrategy(int filteringRate);
}

// Validates method_parameters at query time
public interface SearchParameterValidator {
    /** Returns null if valid, ValidationException otherwise. */
    ValidationException validate(Map<String, Object> params, QueryContext context);

    SearchParameterValidator NOOP = (params, ctx) -> null;
}
```

## Composition on KNNVectorFieldType

These are set at mapping time and read at query time:

```java
public class KNNVectorFieldType extends MappedFieldType {
    // Existing fields...
    SearchCapabilities searchCapabilities;
    SearchStrategyProvider searchStrategyProvider;
    SearchParameterValidator searchParameterValidator;
}
```

## How Algorithms Compose

### Engine-based (FAISS HNSW)

```java
mappedFieldType.searchCapabilities = SearchCapabilities.engineBased();
mappedFieldType.searchStrategyProvider = (filteringRate) -> new KnnSearchStrategy.Hnsw(filteringRate);
mappedFieldType.searchParameterValidator = (params, ctx) ->
    engine.getKNNLibrarySearchContext(method).validate(params, ctx);
```

### Cluster ANN

```java
mappedFieldType.searchCapabilities = SearchCapabilities.noNativeFiles().withRadialSearch();
mappedFieldType.searchStrategyProvider = (filteringRate) -> KnnSearchStrategy.EXACT;
mappedFieldType.searchParameterValidator = SearchParameterValidator.NOOP;
```

### Future algorithm (e.g., IVF-flat)

```java
mappedFieldType.searchCapabilities = SearchCapabilities.noNativeFiles().withRadialSearch();
mappedFieldType.searchStrategyProvider = (filteringRate) -> new KnnSearchStrategy.Ivf(nprobe);
mappedFieldType.searchParameterValidator = IvfParameterValidator.INSTANCE;
```

## Query Path After Refactoring

```java
// KNNQueryBuilder.doToQuery() — replaces knnEngine != UNDEFINED guard
fieldType.getSearchParameterValidator().validate(methodParameters, queryContext);

// KNNWeight.approximateSearch() — replaces EngineLessMethod.isEngineLess() guard
if (engineFiles.isEmpty() && fieldType.getSearchCapabilities().requiresNativeEngineFiles()) {
    return EMPTY_TOPDOCS;
}

// MemoryOptimizedKNNWeight.queryIndex() — replaces hardcoded HNSW strategy
KnnSearchStrategy strategy = fieldType.getSearchStrategyProvider().getSearchStrategy(filteringRate);
KnnCollector collector = collectorManager.newCollector(visitedLimit, strategy, context);

// Radial search check — replaces engine-level check
if (!fieldType.getSearchCapabilities().supportsRadialSearch()) {
    throw new UnsupportedOperationException(...);
}
```

## Benefits

1. **No god interface** — Each piece is independently testable and replaceable.
2. **Mix and match** — Two algorithms can share the same `SearchCapabilities` but differ in strategy.
3. **No inheritance tree** — Just composition of small pieces.
4. **Open for extension** — Adding a new concern (e.g., `WarmupStrategy`) is a new field, not a new method on every implementation.

## Adding a New Engine-Less Algorithm

With this design, adding a new algorithm requires:

1. Add to `EngineLessMethod` enum
2. Compose its `SearchCapabilities`, `SearchStrategyProvider`, and `SearchParameterValidator`
3. Create its mapper and codec

No changes needed in `KNNQueryBuilder`, `KNNWeight`, or `MemoryOptimizedKNNWeight`.

## Migration Path

1. Introduce the three interfaces + common implementations
2. Add fields to `KNNVectorFieldType`, set at mapping time in each mapper
3. Replace the 4 guard points one at a time
4. Remove ad-hoc `EngineLessMethod.isEngineLess()` and `knnEngine != UNDEFINED` checks
