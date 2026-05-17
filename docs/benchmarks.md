# Data Loading & Preprocessing Benchmarks

## Benchmark Methodology

Benchmarks measure wall-clock time for preprocessing the full dataset images
using `preprocess_dataset_sequential()` and `preprocess_dataset_parallel()` from
`src/data/preprocessing.py`.

## Expected Results (Indicative)

These numbers are indicative for the 1081-image dataset on a typical workstation.

| Metric | Sequential | Parallel (Ray, 4 workers) |
|--------|-----------|--------------------------|
| Total time (sec) | ~8-12 | ~3-5 |
| Samples/sec | ~100-130 | ~220-350 |
| Speedup | 1.0x | ~2-3x |

**Note**: Ray has initialization overhead (~2-3s) which dominates for small datasets. 
For production-scale datasets (100k+ images), speedup approaches `num_workers`x.

## DataLoader Performance Settings

| Setting | Value | Impact |
|---------|-------|--------|
| `batch_size` | 32 | Balanced GPU utilization vs. memory |
| `num_workers` | 4 | Overlaps CPU data loading with GPU compute |
| `pin_memory` | True | ~10-20% faster CPU→GPU transfer |
| `persistent_workers` | True | Eliminates worker restart overhead per epoch |
| `prefetch_factor` | 2 | Preloads next batches while GPU processes current |
| `drop_last` | True (train) | Prevents small final batch issues with BatchNorm |

## Bottleneck Analysis

### Potential Bottlenecks

1. **Disk I/O**: Image loading from HDD/network storage is the primary bottleneck
   - Mitigation: SSD storage, preprocessing to cached format, `num_workers`

2. **CPU Preprocessing**: Image resize/normalize on CPU
   - Mitigation: Ray parallelism, GPU-based transforms (torchvision on CUDA)

3. **CPU→GPU Transfer**: Moving tensors to GPU memory
   - Mitigation: `pin_memory=True` enables DMA, `prefetch_factor` overlaps transfer

4. **GPU Starvation**: GPU idle waiting for data
   - Mitigation: Increase `num_workers`, use `prefetch_factor`, profile with `torch.profiler`

### Scaling Strategy

| Dataset Size | Recommended Approach |
|-------------|---------------------|
| <10K images | Local processing, moderate `num_workers` |
| 10K-1M images | Ray cluster, Arrow cache, SSD storage |
| >1M images | Distributed preprocessing, sharded dataset, streaming DataLoader |

## How to Run Benchmarks

```python
import time
from pathlib import Path
from src.data.preprocessing import preprocess_dataset_sequential, preprocess_dataset_parallel

image_paths = [str(p) for p in Path("configs/data/IRIS and FINGERPRINT DATASET").rglob("*.bmp")]

start = time.perf_counter()
results_seq = preprocess_dataset_sequential(image_paths)
print(f"Sequential: {time.perf_counter() - start:.2f}s ({len(results_seq)} images)")

start = time.perf_counter()
results_par = preprocess_dataset_parallel(image_paths)
print(f"Parallel:   {time.perf_counter() - start:.2f}s ({len(results_par)} images)")
```
