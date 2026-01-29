# 🐙 Octopus-Inspired GPU Load Balancing

**Bio-inspired adaptive block assignment for image processing**

---

## TL;DR

I achieved **12x total speedup** over fair GPU baselines by considering the **full cost**: setup time + memory + kernel execution.

| Metric | Grid-Stride (Fair) | Hybrid (Ours) | Improvement |
|--------|-------------------|---------------|-------------|
| Setup time | ~150ms | ~1ms | **148x faster** |
| Memory | 341 MB | 0.03 MB | **11,698x less** |
| Kernel time | ~6ms | ~6ms | ~same |
| **TOTAL** | ~156ms | ~7ms | **12.45x faster** |

**Key insight:** When kernel performance is similar, **setup cost and memory usage determine the winner.**

---

## Contribution

What makes this different from generic "segmented/irregular scheduling":

| Aspect | This Work |
|--------|-----------|
| **Problem** | Ragged 2D stencil (blur) across variable-sized images — not scan/reduce |
| **Cost Model** | **Total cost** = Setup (CPU) + Memory + Kernel — not just kernel throughput |
| **Technique** | O(num_blocks) block metadata vs O(total_pixels) mapping table |
| **Claim** | Kernel throughput is similar; **system-level costs determine the winner** |

The key insight is that for image-aware operations, the "fair" baseline (Grid-Stride with O(1) lookup) requires expensive pre-computation that dominates total runtime.

---

## The Journey: From 252x to Honest 12x

### What I Originally Claimed
> "252x speedup on GPU parallel processing!"

### What I Discovered

| Baseline | Speedup | Problem |
|----------|---------|---------|
| Naive (1 thread/image) | 252x | ❌ Strawman — nobody does this |
| Grid-Stride (O(n) search) | 9-10x | ❌ Unfair — baseline has O(n) bug |
| Grid-Stride (O(1) lookup) | 0.95x | 😬 Kernel-only comparison |
| **Grid-Stride (O(1) + setup)** | **12x** | ✅ **Fair total cost comparison** |

### The Real Win

Grid-Stride with O(1) lookup needs a **huge pre-computed lookup table**:
- `pixel_to_image[total_pixels]` — one entry per pixel
- 100M pixels = **400 MB** of memory
- O(N) time to build

Hybrid only needs **tiny block arrays**:
- `block_to_image[num_blocks]` — one entry per block
- 500 images ≈ 500 blocks = **0.03 MB**
- O(images) time to build

---

## Benchmark Results

### Memory-Aware Benchmark (Main Result)

Comparing **total cost**: Setup + Memory + Kernel

| Test | Pixels | Setup | Memory | Kernel | **TOTAL** |
|------|--------|-------|--------|--------|-----------|
| Flickr Pure | 89M | 210x | 11,545x | 1.00x | **16.68x** |
| Flickr + 4K | 98M | 134x | 11,660x | 0.94x | **11.74x** |
| Flickr + 8K | 123M | 79x | 11,925x | 0.95x | **5.91x** |
| Flickr 1000 | 179M | 124x | 11,574x | 0.96x | **13.78x** |
| Flickr 1000 + 8K | 213M | 193x | 11,787x | 1.00x | **14.13x** |
| **AVERAGE** | — | **148x** | **11,698x** | 0.97x | **12.45x** |

*(Ratios = Grid-Fair / Hybrid, higher = Hybrid wins)*

### Kernel-Only Benchmark

When comparing **only kernel execution time** (ignoring setup):

| Test | Grid-Stride | Hybrid | Ratio |
|------|-------------|--------|-------|
| Flickr Pure | 6.13ms | 6.46ms | 0.95x |
| Flickr + 4K | 6.79ms | 6.84ms | 0.99x |
| Flickr + 8K | 8.12ms | 8.55ms | 0.95x |

**Conclusion:** Kernel performance is similar. The win comes from setup + memory.

---

## When Does Hybrid Win?

### ✅ Hybrid wins (use it):

| Scenario | Why |
|----------|-----|
| **New batches each time** | Setup cost matters (12x faster) |
| **Memory-constrained devices** | 11,000x less memory (Jetson, mobile) |
| **Streaming/real-time** | Can't afford 150ms setup delay |
| **Variable-size images** | Block-per-image fails on large images |

### ⚠️ Similar performance:

| Scenario | Why |
|----------|-----|
| **Same batch repeated 100+ times** | Setup cost amortized |
| **Kernel-only comparison** | Both achieve similar throughput |

### ❌ Don't use Hybrid:

| Scenario | Why |
|----------|-----|
| **Per-pixel independent ops** | Grid-stride is simpler, equally fast |
| **Already balanced workload** | No imbalance to solve |

---

## The Octopus Insight

An octopus has ~500 million neurons distributed across 8 arms. Each arm operates semi-independently, yet they coordinate perfectly.

**How?** The octopus pre-computes force distribution so no arm waits for another.

**GPU translation:** Pre-compute work distribution so no thread waits — but do it **efficiently**.

```
Grid-Stride-Fair: Pre-compute per-PIXEL  → O(N) setup, O(N) memory
Hybrid:           Pre-compute per-BLOCK  → O(B) setup, O(B) memory
                  where B << N
```

---

## Implementation

### Core Algorithm (~30 lines)

```python
def compute_hybrid_assignment(sizes, threshold=65536):
    """
    Adaptive block assignment:
    - Small images: 1 block (locality)
    - Large images: subdivide (load balance)
    """
    block_to_image = []
    block_start = []
    block_end = []
    
    for img_id, size in enumerate(sizes):
        if size <= threshold:
            # Small image: 1 block
            block_to_image.append(img_id)
            block_start.append(0)
            block_end.append(size)
        else:
            # Large image: subdivide
            num_blocks = ceil(size / threshold)
            for b in range(num_blocks):
                block_to_image.append(img_id)
                block_start.append(b * threshold)
                block_end.append(min((b+1) * threshold, size))
    
    return block_to_image, block_start, block_end
```

### GPU Kernel

```python
@cuda.jit
def hybrid_kernel(images_flat, offsets, widths, heights,
                  block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    
    # O(1) lookup — no search!
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    # Image info
    offset = offsets[img_id]
    w = widths[img_id]
    
    # Threads cooperate within block's range
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        # ... process pixel with image context
```

---

## Files

| File | Description |
|------|-------------|
| `memory_benchmark.py` | **Main benchmark** — Total cost comparison |
| `hybrid_benchmark.py` | Kernel comparison (Hybrid vs Grid-Stride-Search) |
| `fair_hybrid_benchmark.py` | Kernel comparison (Hybrid vs Grid-Stride-Fair) |
| `medical_benchmark.py` | Medical imaging tests |

---

## Quick Start

```bash
git clone https://github.com/matthewlam721/octopus-parallel.git
cd octopus-parallel

pip install numba numpy scipy pillow

# Download Flickr8k from Kaggle
# Place in ./Images/

# Run main benchmark
python memory_benchmark.py
```

---

## What I Learned

### 1. Fair baselines matter
My initial 252x was against a strawman. Real contribution is 12x vs fair baseline.

### 2. Total cost matters
Kernel time alone is misleading. Setup + memory + kernel = true comparison.

### 3. Know when your approach wins
- ✅ New batches, memory-constrained: Hybrid wins
- ⚠️ Repeated batches, kernel-only: Similar performance

### 4. Simple isn't always optimal
Block-per-image is simple but fails on imbalanced workloads. Hybrid adds minimal complexity but handles all scenarios.

---

## Future Work

- [ ] Edge deployment (NVIDIA Jetson) — where memory savings matter most
- [ ] Real algorithms (U-Net, segmentation)
- [ ] Video processing datasets
- [ ] Framework integration (PyTorch, JAX)

---

## Conclusion

**The octopus doesn't waste energy computing per-neuron lookup tables. Neither should your GPU.**

For image-aware operations with variable-sized workloads:
- **12x faster** total time
- **11,000x less** memory
- **Zero runtime overhead** after setup

The key insight: **efficiency of the pre-computation matters as much as the kernel itself.**

---

**Author:** Matthew, UIUC MCS  
**Contact:** matthewlam721@gmail.com  
**Repo:** [github.com/matthewlam721/octopus-parallel](https://github.com/matthewlam721/octopus-parallel)

---

## Appendix: Full Benchmark Output

```
======================================================================
MEMORY-AWARE BENCHMARK SUMMARY
======================================================================

  Test                       Pixels      Setup     Memory     Kernel      TOTAL
  ---------------------------------------------------------------------------
  Flickr Pure           89,416,278       210x     11545x      1.00x     16.68x
  Flickr + 4K           97,710,678       134x     11660x      0.94x     11.74x
  Flickr + 8K          122,593,878        79x     11925x      0.95x      5.91x
  Flickr 1000          179,455,121       124x     11574x      0.96x     13.78x
  Flickr 1000 + 8K     212,632,721       193x     11787x      1.00x     14.13x

  AVERAGE                         -       148x     11698x      0.97x     12.45x

======================================================================
KEY FINDINGS
======================================================================

  1. SETUP TIME: Hybrid is 148x faster
  2. MEMORY: Hybrid uses 11,698x less memory  
  3. KERNEL: Similar performance
  4. TOTAL: Hybrid is 12.45x faster overall

  🐙 HYBRID WINS when considering TOTAL cost!

======================================================================
```

*Tested on NVIDIA RTX 4090, January 2026*
