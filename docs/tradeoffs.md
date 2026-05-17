# Trade-offs & Design Rationale

## Technology Choices

### Why Ray for Parallel Preprocessing?

| Pros | Cons |
|------|------|
| Easy Python-native parallelism | Added dependency (~200MB) |
| Scales from local to multi-node clusters | Overhead for small datasets (<1000 images) |
| Built-in fault tolerance | Learning curve for advanced features |
| Active ecosystem, good PyTorch integration | |

**Decision**: Ray provides a production-grade parallelism framework that scales from a single machine to multi-node clusters. For this small dataset, `multiprocessing.Pool` would suffice, but Ray enables horizontal scaling without code changes when data volumes grow.

**Fallback**: The code gracefully falls back to sequential processing if Ray is unavailable.

### Why PyArrow for Metadata?

| Pros | Cons |
|------|------|
| Columnar format - fast reads and filters | Serialization overhead for tiny datasets |
| Zero-copy reads in many cases | Additional dependency |
| Parquet persistence for cache reuse | Overkill for 1081 files |
| Interoperable with Pandas, Spark, etc. | |

**Decision**: Arrow metadata tables provide fast reads and filters. At production scale (millions of images from camera/radar sensors), they avoid expensive filesystem or cloud storage listing operations.

### Why Simple CNN Instead of ResNet/ViT?

| Pros | Cons |
|------|------|
| Fast to train, easy to debug | Lower accuracy ceiling |
| Clear architecture - nothing to hide | Less "impressive" on paper |
| Focus stays on infrastructure | |

**Decision**: The focus is on pipeline correctness and extensibility over raw model accuracy. A simple CNN validates the end-to-end pipeline; swapping to ResNet is a one-line config change.

### Why Hydra for Configuration?

| Pros | Cons |
|------|------|
| Composable YAML configs | Slightly magical (auto-instantiation) |
| CLI override support | Working directory changes by default |
| Multi-run support for sweeps | |

**Decision**: Hydra is industry-standard for ML configs. Eliminates hardcoded values and enables reproducible experiments via config snapshots.

### Why Gated Attention Fusion?

| Pros | Cons |
|------|------|
| Learns per-sample modality importance | Slightly more parameters than concat |
| Outperforms naive concatenation | Gate weights not directly interpretable |
| Softmax gate ensures stable gradients | |

**Decision**: The gated attention fusion learns a 2-element softmax weight vector per sample, deciding how much to trust iris vs. fingerprint. This is the right inductive bias for biometrics - image quality varies per sample, so a fixed 50/50 split is suboptimal. The concat baseline is retained in the registry for comparison.

**Result**: Both strategies are available via `model.fusion.strategy=attention|concat`.

## Architecture Trade-offs

### Lazy Loading vs. Pre-loading

**Chosen**: Lazy loading (images loaded in `__getitem__`)

- Pro: Low memory footprint, supports datasets larger than RAM
- Con: I/O becomes a bottleneck if disk is slow
- Mitigation: `num_workers` in DataLoader overlaps I/O with computation

### Single-process vs. Distributed Training

**Chosen**: Single-process training

- Pro: Simpler, debuggable, sufficient for this dataset
- Con: Doesn't demonstrate DDP
- Note: The architecture supports wrapping with `DistributedDataParallel` without code changes to the model or dataset

### Checkpointing Strategy

**Chosen**: Save every N epochs + best model + final epoch

- `best_model.pt` is saved whenever validation loss improves
- Periodic checkpoints every N epochs for resumability
- Early stopping halts training after `patience` epochs without improvement
- Pro: Never misses the best model; prevents wasted compute on overfitting
