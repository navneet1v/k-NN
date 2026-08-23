
# Locality-Aware Physical Layout for HNSW Graphs
## Design Proposal (Algorithm Only)

> This document intentionally focuses only on the core layout algorithm. It omits implementation details, pseudocode, Lucene internals, evaluation, and prior work.

---

# Design Philosophy

The central observation behind this proposal is that the HNSW graph already encodes the relationships needed to build an efficient physical storage layout. Rather than imposing an external partitioning algorithm, the storage layout should emerge from the graph's own structure.

The objective is **not** to produce the best graph partition. Instead, the objective is to maximize **traversal locality**. If a search naturally walks between a group of nodes, those nodes should ideally reside within the same disk page.

Several principles guide the design:

- Physical locality should follow traversal locality rather than insertion order.
- Disk pages should represent coherent graph neighborhoods instead of arbitrary balanced partitions.
- Limited replication is preferable to frequent page crossings.
- Structural properties of the graph should be sufficient; no runtime query profiling should be required.

The resulting layout is intended to preserve HNSW search behavior while reducing the number of storage pages touched during graph traversal.

---

# Greedy Page Growing

The proposed layout is built incrementally instead of through global partitioning.

Each page begins from a carefully selected seed. Rather than assigning the entire graph at once, the page grows outward by repeatedly absorbing nearby nodes that exhibit the strongest structural affinity with the nodes already assigned to that page.

This process is intentionally local. HNSW search itself is a greedy local walk, so constructing pages using local expansion naturally aligns storage organization with expected search behavior.

As a page grows, the neighborhood gradually stabilizes around a coherent region of the graph. Dense clusters are likely to remain together, while loosely connected bridges naturally become page boundaries.

The quality of a page depends on three competing forces:

- Maintaining strong internal connectivity.
- Minimizing edges leaving the page.
- Respecting the fixed storage capacity of each page.

The algorithm does not attempt to find a globally optimal partition. Instead, it relies on many locally optimal growth decisions that collectively produce a highly localized storage layout. This makes the approach scalable while remaining faithful to the underlying graph structure.

Page size also becomes an important design parameter. Smaller pages reduce wasted I/O but increase boundary crossings. Larger pages improve locality but may increase unnecessary data reads. The algorithm is intentionally independent of page size so that different storage systems can tune this parameter independently.

---

# Hub Detection

Not all nodes play the same role within an HNSW graph. Some naturally act as routing points and are encountered more frequently during search.

These routing nodes, referred to here as hubs, should not be viewed as independent storage objects. Instead, they should be viewed as indicators of where graph neighborhoods begin.

Several structural properties can indicate hubness, including higher HNSW levels, large incoming connectivity, or broader influence across neighboring regions. The exact definition is intentionally left flexible because different datasets may benefit from different structural signals.

The important design decision is that hubs are used to improve page formation rather than becoming dedicated hub pages.

Creating pages composed primarily of hubs would concentrate traffic onto a small number of storage pages, increasing contention and defeating the purpose of improving locality.

Instead, hubs act as anchors around which nearby regions are organized. They influence where pages begin, but the page itself ultimately contains the surrounding neighborhood rather than only globally important nodes.

This distinction allows routing information to shape the layout without creating artificial bottlenecks.

---

# Boundary Replication

No matter how carefully pages are constructed, some nodes naturally belong to multiple neighboring regions.

These boundary nodes create repeated page transitions because searches approaching from different directions must repeatedly cross page boundaries to reach them.

Rather than forcing every node to have a single physical location, this design allows a small number of strategically selected boundary nodes to be replicated.

Replication is not intended to improve graph quality. The graph itself remains unchanged. Instead, replication reduces storage-level boundary crossings by allowing neighboring pages to contain local copies of important connecting nodes.

Replication should occur only after the primary layout has been established. Before pages exist there are no boundaries to optimize. Once the layout is complete, nodes with disproportionate influence across page boundaries can be identified and selectively duplicated.

The replication budget should remain intentionally small. Excessive replication wastes storage and complicates maintenance, while insufficient replication leaves many avoidable page crossings. The objective is to capture the majority of locality benefits using only a very small fraction of duplicated nodes.

This philosophy is inspired by locality-preserving storage systems where modest redundancy is exchanged for significantly lower access costs.

---

# Putting Everything Together

The complete design can be viewed as a sequence of progressively refined decisions.

The graph is first analyzed to identify structural signals that characterize local neighborhoods and important routing nodes.

Those signals guide the selection of seeds that initiate page construction.

Each page then expands greedily into its surrounding neighborhood until the page reaches its capacity, allowing local graph structure to determine the physical organization.

Once all primary pages have been formed, the layout is examined to identify boundary nodes whose duplication would substantially reduce future page crossings.

Finally, the completed pages, together with the limited replicated boundary nodes, are serialized into their physical storage layout.

The resulting organization is not a graph partition in the traditional sense. It is a storage layout derived from graph structure whose primary objective is to maximize traversal locality while minimizing storage-level read amplification.

The central philosophy is simple:

> The HNSW graph already contains the information needed to organize storage efficiently. The storage system should follow the graph rather than forcing the graph to follow the storage system.
