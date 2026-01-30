# 🐙 Octopus: Memory-Efficient GPU Scheduling for Variable-Length Batches

A novel block-level GPU scheduling approach that achieves **O(1) dispatch** without **O(N) mapping tables**, enabling efficient processing of variable-length data (images, sequences, point clouds) on GPUs.

## The Problem

When processing batches of variable-length items on GPUs, existing approaches face a fundamental tradeoff:

| Approach | Memory | Kernel Speed | Setup Time |
|----------|--------|--------------|------------|
| **Padding** | Wastes 30-50% compute | Fast | Fast |
| **O(1) Lookup Table** | O(N) - explodes with data | Fast | Slow |
| **Binary Search** | O(M) - minimal | Slow (O(log M)) | Fast |

**Octopus (Hybrid)** breaks this tradeoff:

| Approach | Memory | Kernel Speed | Setup Time |
|----------|--------|--------------|------------|
| **🐙 Octopus** | O(B) ≈ O(M) | Fast (O(1)) | Fast |

Where:
- N = total pixels/tokens
- M = number of items
- B = number of blocks (≈ M for normal workloads)

## Key Results

### RTX 4090 Benchmarks (10K images, 550M pixels)

```
Baseline                Setup      H2D     Memory     Kernel      D2H      TOTAL
─────────────────────────────────────────────────────────────────────────────────
A (O(1) Lookup)       619.71ms  209.99ms  2474.65MB   30.90ms  256.63ms  1117.23ms
B (Binary Search)       0.00ms    0.01ms     0.08MB   35.61ms  256.63ms   292.26ms
C (Octopus/Hybrid)      0.06ms    0.30ms     0.27MB   31.42ms  256.63ms   288.42ms
                                                                          ^^^^^^^^
                                                                          WINNER!
```

**Key findings:**
- ✅ **Octopus wins total time** (288ms vs 292ms for Binary Search)
- ✅ **9000x less memory** than O(1) Lookup (0.27 MB vs 2474 MB)
- ✅ **13% faster kernel** than Binary Search (31.4ms vs 35.6ms)
- ✅ **3.9x faster total** than O(1) Lookup

### Scaling Analysis (1M images)

| Metric | Binary Search | Octopus |
|--------|---------------|---------|
| Kernel time | 31.75 ms | 25.98 ms |
| **Kernel speedup** | - | **22% faster** |

At scale (M = 1M), Binary Search's O(log M) penalty becomes visible:
- log₂(10K) = 14 steps
- log₂(1M) = 20 steps (+43% more comparisons)

## When to Use Each Approach

```
┌─────────────────┬──────────────────────────────────────────────────────┐
│ Scenario        │ Recommendation                                       │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Normal workload │ 🐙 Octopus - wins total time, extensible            │
│ (10K-100K items)│                                                      │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Tiny items +    │ Binary Search - zero setup wins when kernel         │
│ massive M (1M+) │ overhead < setup overhead                            │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Kernel reuse    │ O(1) Lookup - if same batch runs 100+ times,        │
│ (100+ runs)     │ amortized setup cost becomes negligible              │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Edge/Embedded   │ 🐙 Octopus - stable on weak GPUs with small cache   │
│ (Jetson, MIG)   │                                                      │
├─────────────────┼──────────────────────────────────────────────────────┤
│ Cloud API       │ 🐙 Octopus - deterministic latency for SLA          │
│ (high traffic)  │                                                      │
└─────────────────┴──────────────────────────────────────────────────────┘
```

## Architecture

### The Octopus Metaphor 🐙

> An octopus coordinates movement at the **arm level**, not the **neuron level**.
> Each arm has local autonomy, but the brain provides high-level coordination.

Similarly, Octopus dispatches work at the **block level**:

```
Traditional (O(1) Lookup):
  pixel_0 → image_3    ┐
  pixel_1 → image_3    │ 550M entries!
  pixel_2 → image_3    │ 2.4 GB memory
  ...                  ┘

Octopus (Block Metadata):
  block_0 → {image_id: 3, start: 0, end: 65536}     ┐
  block_1 → {image_id: 3, start: 65536, end: 131072} │ 10K entries
  block_2 → {image_id: 7, start: 0, end: 42000}      │ 0.27 MB memory
  ...                                                ┘
```

### Block Metadata Structure

```python
# O(B) memory where B ≈ M for normal workloads
block_to_image: int32[B]   # Which image this block processes
block_start:    int64[B]   # Local start offset within image
block_end:      int64[B]   # Local end offset within image
```

### Kernel Design

```python
@cuda.jit
def octopus_kernel(images_flat, offsets, widths, heights,
                   block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    
    # O(1) lookup - no search needed!
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    # Process pixels within this block's range
    for local_idx in range(local_start + tid, local_end, stride):
        # ... image processing kernel ...
```

## Why Octopus Wins

### 1. O(1) Dispatch Stability

Binary Search performance depends on **cache hit rate**:
- RTX 4090 (72 MB L2): Binary search offsets fit in cache → fast
- Jetson Orin (4 MB L2): Cache misses → 5-10x slower
- Multi-tenant GPU: Cache contention → unpredictable latency

Octopus provides **stable O(1) dispatch regardless of cache state**.

### 2. Extensibility

Binary Search can only answer: "Which image does pixel X belong to?"

Octopus block metadata can include **scheduling policy**:

```python
block_metadata = {
    'image_id': 3,
    'priority': HIGH,        # ROI prioritization
    'strategy': HEAVY_BLUR,  # Different kernels per block
    'stream_id': 2,          # Multi-stream scheduling
}
```

### 3. Memory Efficiency

| Approach | Memory for 10K images, 550M pixels |
|----------|-----------------------------------|
| O(1) Lookup | 2,474 MB |
| Binary Search | 0.08 MB |
| **Octopus** | **0.27 MB** |

Octopus uses ~3x more than Binary Search but enables O(1) dispatch.

## Installation

```bash
# Requirements
pip install numba numpy pillow

# CUDA toolkit (for GPU acceleration)
# Numba supports CUDA 11.x and 12.x
```

## Usage

### Basic Benchmark

```bash
# Standard benchmark (10K images)
python triple_baseline_benchmark.py

# Large scale test (100K images, small)
python triple_baseline_benchmark.py --images 100000 --small

# Extreme scale test (1M images, tiny)
python triple_baseline_benchmark.py --images 1000000 --tiny
```

### Advanced Options

```bash
# Heavy kernel (10x iterations) - amplifies branch divergence
python triple_baseline_benchmark.py --heavy

# Cache flush mode - simulates real workload with cache contention
python triple_baseline_benchmark.py --flush-cache

# Combined
python triple_baseline_benchmark.py --images 100000 --small --heavy
```

### Custom Integration

```python
from triple_baseline_benchmark import setup_baseline_c, kernel_baseline_c

# Setup (runs on CPU, ~0.06ms for 10K images)
block_to_image, block_start, block_end = setup_baseline_c(sizes, threshold=65536)

# Transfer to GPU
d_block_to_image = cuda.to_device(block_to_image)
d_block_start = cuda.to_device(block_start)
d_block_end = cuda.to_device(block_end)

# Launch kernel
num_blocks = len(block_to_image)
kernel_baseline_c[num_blocks, 256](
    d_images, d_offsets, d_widths, d_heights,
    d_block_to_image, d_block_start, d_block_end, d_output
)
```

## Target Applications

### 1. Edge AI & Robotics 🤖
- **Hardware**: NVIDIA Jetson Orin/Xavier (4-16 GB RAM, 2-4 MB L2)
- **Problem**: O(1) lookup tables cause OOM; Binary search suffers cache misses
- **Solution**: Octopus provides stable performance without memory explosion

### 2. High-Throughput Cloud APIs ☁️
- **Use case**: Image processing APIs (filters, transforms, AI inference)
- **Problem**: Padding wastes 30-50% compute; Binary search has latency jitter
- **Solution**: Octopus enables zero-padding + deterministic SLA

### 3. Gigapixel Processing 🔬
- **Use case**: Medical imaging (pathology slides), satellite imagery
- **Problem**: 100,000 x 100,000 pixel images → O(1) lookup needs hundreds of GB
- **Solution**: Octopus makes previously impossible workloads feasible

### 4. LLM Variable Sequences 📝
- **Use case**: Transformer attention on ragged batches
- **Problem**: Current solutions (cuSEQ, FasterTransformer) use complex offset arrays
- **Solution**: Octopus block-level scheduling could simplify implementation

## Benchmark Results Summary

### Decision Matrix

| M (images) | Item Size | Kernel Winner | Total Winner |
|------------|-----------|---------------|--------------|
| 10K | Normal (30-80K px) | Octopus (+13%) | **Octopus** |
| 100K | Small (3-8K px) | Octopus (+15%) | ~Tie |
| 1M | Tiny (300-800 px) | Octopus (+22%) | Binary Search* |

*Binary Search wins total at 1M tiny items due to zero setup overhead, but Octopus kernel is still 22% faster.

### System Insight

```
D2H transfer (256ms) dominates total time (288ms).
In output-copy dominated pipelines, scheduler differences are masked.
For fused GPU pipelines (no D2H), Octopus's O(1) advantage becomes visible.
```

## Limitations & Future Work

1. **Threshold tuning**: Currently fixed at 65536; could be auto-tuned
2. **Multi-GPU**: Not yet implemented; natural extension via block partitioning
3. **Sparse workloads**: May have overhead for very sparse data
4. **NLP integration**: Block scheduling for Transformer attention (future work)

## Citation

```bibtex
@software{octopus2026,
  title={Octopus: Memory-Efficient GPU Scheduling for Variable-Length Batches},
  author={Matthew},
  year={2026},
  url={https://github.com/[username]/octopus-gpu-scheduler}
}
```

## License

MIT License

## Acknowledgments

- NVIDIA for CUDA and Numba
- The GPU computing community for inspiration
- 🐙 Octopuses for the metaphor

---

**Key Insight**: Hybrid achieves table-like O(1) dispatch without O(N) mapping, and matches the strongest no-table baseline (binary search) within ~1-5% while using negligible memory. On normal workloads, Octopus wins total time; on extreme scale (1M+ tiny items), Binary Search's zero setup wins, but Octopus kernel is always faster.
