# 🐙 Octopus: Block-Level GPU Scheduling for Variable-Length Batches

I needed to process batches of variable-sized images on GPU without padding. Tried three approaches, benchmarked them properly, and found some surprising results.

Why "Octopus"? Each CUDA block independently knows its task via O(1) lookup, like how octopus arms process locally without waiting for the brain. That's the extent of the analogy—the rest is just benchmarks.

## The Problem

You have 10,000 images of different sizes. You want to run a kernel on all pixels. Options:

1. **Pad everything** to max size → wastes compute
2. **Flatten into one array** → but then how does each thread know which image it's processing?

I tested three solutions to #2.

## Three Approaches

```
Flattened pixels:  [████ img0 ████|██ img1 ██|██████ img2 ██████|...]
                    ↑ pixel 12345 belongs to which image?

A (Table):    pixel_to_image = [0,0,0,0,0, 1,1,1, 2,2,2,2,2,2...]
              └─ 500M entries = 2GB memory ❌

B (Search):   offsets = [0, 50000, 80000, 140000...]
              └─ Binary search per pixel, O(log M) ⚠️
              └─ 0.08 MB, but cache-dependent

C (Block):    block_to_image = [0, 0, 1, 2, 2, 2...]
              block_range    = [(0,32K), (32K,50K), (0,30K), ...]
              └─ O(1) lookup per block ✅
              └─ 0.27 MB, deterministic
```

### A: Lookup Table
```python
# Build a mapping for every pixel
pixel_to_image[pixel_idx] → image_id

# 500M pixels × 4 bytes = 2GB memory
```

### B: Binary Search  
```python
# Store only offsets (where each image starts)
# Kernel does binary search: O(log M) per pixel
offsets = [0, 50000, 120000, ...]  # M entries
```

### C: Block-Level Metadata
```python
# Each CUDA block knows which image it handles
# O(1) lookup per block, not per pixel
block_to_image[block_id] → image_id
```

## Results (RTX 4090, 10K images, ~500M pixels)

| Approach | Memory | Kernel | Total |
|----------|--------|--------|-------|
| A (Table) | 2475 MB | 31 ms | 1117 ms |
| B (Search) | 0.08 MB | 36 ms | 292 ms |
| C (Block) | 0.27 MB | 31 ms | 288 ms |

**Findings:**

- A is useless. Setup (619ms) + H2D transfer (210ms) kills it.
- B and C are close. C wins by 4ms (1.4%).
- C kernel matches A's speed while using **9000x less memory** than A.
- C uses only **0.27 MB** vs A's 2475 MB—this is the main win.

## The Surprising Part

I expected binary search to be way slower due to O(log M) overhead and branch divergence. It wasn't. On RTX 4090, the 0.08 MB offset array fits entirely in L2 cache (72 MB), so binary search is nearly free.

**When I scaled to 1M images:**

| Approach | Kernel | Total |
|----------|--------|-------|
| B (Search) | 31.6 ms | 250 ms |
| C (Block) | 29.3 ms | 254 ms |

B wins total by 1.5%, but C kernel is 8% faster. The log₂(M) penalty is starting to show.

## When Does Each Win?

| Scenario | Winner | Why |
|----------|--------|-----|
| Normal workload (10K images) | C | Fastest total |
| Tiny items, massive M (1M+) | B | Zero setup overhead |
| Weak GPU (small L2 cache) | C | B will cache-miss |
| Need different kernels per block | C | B can't express this |

## What C Can Do That B Can't

Binary search gives you an image ID. That's it.

Block metadata can carry whatever you want:

```python
block_info = {
    'image_id': 3,
    'priority': HIGH,      # process important images first  
    'kernel_type': BLUR,   # different operations per block
    'stream_id': 2,        # multi-stream scheduling
}
```

If you just need image IDs, use B. If you need scheduling flexibility, C is the only option.

## Limitations

- Only tested on RTX 4090. Results will differ on weaker GPUs.
- D2H transfer (257ms) dominates total time, masking kernel differences.
- In fused pipelines (no D2H), the kernel speed difference would matter more.

## Running the Benchmark

```bash
# Basic test
python3 triple_baseline_benchmark.py --images 10000

# Scale test  
python3 triple_baseline_benchmark.py --images 1000000 --tiny

# Stress test (10x compute per pixel)
python3 triple_baseline_benchmark.py --heavy
```

## Code

The benchmark is ~800 lines of Python/Numba CUDA. Nothing fancy. Setup uses `@njit` for speed.

Key kernel structure for approach C:

```python
@cuda.jit
def kernel_c(images, offsets, widths, heights,
             block_to_image, block_start, block_end, output):
    
    block_id = cuda.blockIdx.x
    img_id = block_to_image[block_id]  # O(1)
    
    # Each thread processes pixels in its assigned range
    for local_idx in range(block_start[block_id] + tid, 
                           block_end[block_id], stride):
        # ... do work
```

## Conclusions

1. For variable-length GPU batching, don't use lookup tables. The memory and setup cost isn't worth it.

2. Binary search is surprisingly competitive on modern GPUs thanks to large L2 caches.

3. Block-level metadata gives you similar performance with more flexibility.

4. The "best" approach depends on your workload. I've provided the numbers; pick what fits.

## Where This Actually Matters

On RTX 4090, B ≈ C. But the gap widens on:

- **Edge devices (Jetson)**: 2MB L2 cache means binary search will miss. Block metadata stays O(1).
- **Multi-tenant GPUs (MIG)**: Shared cache = contention. Deterministic O(1) beats variable binary search.
- **Gigapixel images (medical/satellite)**: 100K×100K image = 40GB lookup table. Block metadata = only option.

I only have a 4090 to test on. If someone has Jetson results, I'd love to see them.

---

Questions or benchmarks on other hardware welcome.
