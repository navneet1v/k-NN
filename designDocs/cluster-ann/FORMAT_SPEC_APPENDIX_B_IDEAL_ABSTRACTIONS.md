# Appendix B — Ideal abstractions (read/write twins)

> **Staging note.** This is destined for the *ClusterANN Read-Write Format
> Specification* in Pippin, as a new appendix after `Appendix A — Critical
> Decision points`. The existing `B Action Items` heading there will want
> relabelling to `C`.

This belongs in the format spec rather than in the reader or writer design
because the format *is* the contract both sides implement. Neither doc can own
the pairing alone: the reader design describes half, the writer design the other
half, and nothing holds them to being the same shape. This appendix names the
twins and states what keeps them true.

It also discharges the **Extensible** tenet on the code side. "Clear, bounded
extension points" in the layout are only useful if both sides have an extension
point at the *same* boundary.

## B.1 The twins

| format construct | reader side | writer side | shared type |
| --- | --- | --- | --- |
| a field's clusters | `Clusters` | `ClustersWriter` | `.clam` field record |
| one cluster's posting | `Cluster` | `ClusterWriter` | `PostingHeader` |
| the vectors in a posting | `PostingScorer` — iterate + score | `PostingEncoder` — iterate + quantize + pack | `BlockLayout` |
| block positioning | `BlockReader` | `BlockWriter` | `BlockLayout` |
| storage family selection | `ClusterFactory` | `ClusterWriterFactory` | `.clam` `quantizerId` + `docBits` |
| rotation | `prepareQuery(q)` | `prepareVector(v)` | `.clar` |
| centroid geometry | centroid values cursor | `CentroidsWriter` | `CentroidRow` |

## B.2 Three techniques, in order of what they buy

**1. Share the carrier type.** If the reader returns `T` and the writer accepts
the same `T`, the inverse is structural — nobody has to be told. One record per
format construct, describing it *completely*:

```java
// .clap — everything in a posting before its blocks
record PostingHeader(int[] ordinals, long[] soarBits, float[] sortedDistances) { }

// .clac region 2 — one centroid row
record CentroidRow(float[] vector, float normSq, float maxDistance) { }

PostingHeader read (IndexInput  in,  long offset, int clusterSize);
long          write(IndexOutput out, PostingHeader header);
```

The record must be complete even where one side ignores a field. The reader may
*skip* `soarBits` on disk for speed — but a field absent from the record is a
field the writer can change unnoticed.

**2. Share the coordinate system.** Both sides agree that block `b` covers
positions `[b·BS, min((b+1)·BS, clusterSize))`. Duplicated layout arithmetic is
how a reader and writer drift, so it exists once:

```java
interface BlockLayout {
    int blockSize();
    int numBlocks();
    int blockStart(int block);
    int blockLength(int block);
}

interface BlockReader extends BlockLayout, Closeable {
    boolean seekToBlock(int block) throws IOException;   // random access
    void    prefetchBlock(int block) throws IOException;
}

interface BlockWriter extends BlockLayout, Closeable {
    void appendBlock(int length) throws IOException;     // append only
}
```

**3. Identical stem, direction in the suffix.** `Block…`, `Cluster…`,
`Posting…` name the responsibility; `Reader`/`Writer`, `Scorer`/`Encoder` name
the direction. Keep the stem identical even when suffixes don't rhyme — don't
rename `PostingScorer` to `PostingReader` for symmetry, because `Scorer` is
accurate about what it fuses.

## B.3 The storage family, both directions

The layout of `.clap` and the code that produces or consumes it are inseparable:
the corrections sit beside the codes, and scoring straight out of them is the
point. So the extension point is the *whole* cluster implementation on both
sides, selected from the same `.clam` fields:

```java
// read
interface Cluster        { PostingScorer scorer(ScanParams p, Bits wanted); }
interface PostingScorer  { boolean advance(float minCompetitive); int ord(); float score(); }

// write
interface ClusterWriter  { PostingEncoder encoder(BuildParams p); }
interface PostingEncoder { void add(int ord, float[] vector); long finish(); }
```

A second quantization family (PQ, binary) adds one implementation on each side
and touches nothing above them. **A pluggable quantizer inside a fixed posting
writer does not hold this seam** — a family whose corrections are not
`lower | upper | add | sum` breaks both the carrier and the packing.

Rotation belongs inside the family, not above it. `.clar` exists only to serve
the quantizer, so the family owns writing and applying it — `prepareVector` on
the write side opposite `prepareQuery` on the read side. Orchestration then
never branches on metric.

## B.4 What does not invert — do not force it

| pair | relationship |
| --- | --- |
| `PostingHeader` read / write | **true inverse**, round-trippable |
| `BlockReader` / `BlockWriter` | inverse in *coordinates* only — reading is random-access, writing is append-only |
| `ClusterFactory` / `ClusterWriterFactory` | **true inverse** — same `.clam` fields in, family impl out |
| `Cluster` / `ClusterWriter` | **true inverse** at "owns one cluster's posting end to end" |
| `BlockScorer` / `BlockEncoder` | **not inverses.** The reader never decodes — it scores straight from the codes. Inventing a `BlockDecoder` for symmetry would misdescribe the design |

Siblings under the same factory, sharing the same layout, is the correct
relationship for the last row. Not mirror images.

## B.5 What keeps it true

Names and hierarchies help humans; only a test stops drift. One property test
per true-inverse pair:

```java
assertEquals(header, read(write(header)));
```

A pair that cannot be written as a round trip is a pair that is not an inverse —
which is the cheapest way to find out that it belongs in B.4 instead of B.1.

---

## Open points before this lands

1. **`ClustersWriter` is the least settled name.** The writer's existing
   `PostingsWriter` occupies that role but is named after the file rather than
   the responsibility. Renaming makes the twin visible; keeping it leaves the
   first row of B.1 with no visible pairing.

2. **B.3 is prescriptive, not descriptive, for one component.** The writer
   design currently has `QuantizerFactory` selecting only a quantizer inside a
   fixed `PostingsWriter` layout. That is the coupling this appendix argues
   against, so if the writer is not going to change, the appendix should say so
   explicitly rather than read as a description of what exists.
