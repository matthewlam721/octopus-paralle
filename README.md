# 🐙 Octopus: Block-Level GPU Scheduling for Variable-Length Batches

I needed to process batches of variable-sized images on GPU without padding. Tried three approaches, benchmarked them on two GPUs, and found that cache size matters more than I expected.

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

## Results

### The Short Version

On a beefy GPU (RTX 4090 with 72MB L2 cache), binary search and block metadata perform about the same. The offset array fits in cache, so the O(log n) lookups are basically free.

On a smaller GPU (T4 with 4MB L2 cache), block metadata wins by 22-28%. Once the offset array spills out of cache, binary search starts paying for those random memory accesses.

### T4 GPU Benchmarks (4MB L2 Cache)

This is where it gets interesting. T4 has similar cache to edge devices like Jetson.

**Synthetic Benchmark:**

| Images | B (Search) Kernel | C (Block) Kernel | Speedup |
|--------|-------------------|------------------|---------|
| 100K | 19.11 ms | 14.89 ms | **22% faster** |
| 200K | 39.79 ms | 29.41 ms | **26% faster** |
| 500K | 102.67 ms | 74.03 ms | **28% faster** |

**Video Frame Benchmark (600 frames, 277M pixels, 16x size imbalance):**

| Approach | Memory | Kernel | Total |
|----------|--------|--------|-------|
| A (Table) | 1059 MB | 81.04 ms | 1446 ms |
| B (Search) | 0.005 MB | 87.82 ms | 515 ms |
| C (Block) | 0.088 MB | 68.37 ms | 497 ms |

**C kernel 22% faster than B on real video data.**

### RTX 4090 Benchmarks (72MB L2 Cache)

For comparison, here's what happens on a high-end GPU where cache isn't a bottleneck.

**Synthetic Benchmark (10K images, 500M pixels):**

| Approach | Memory | Kernel | Total |
|----------|--------|--------|-------|
| A (Table) | 2475 MB | 31 ms | 1117 ms |
| B (Search) | 0.08 MB | 36 ms | 292 ms |
| C (Block) | 0.27 MB | 31 ms | 288 ms |

**Video Frame Benchmark (1300 frames, 862M pixels):**

| Approach | Memory | Kernel | Total |
|----------|--------|--------|-------|
| A (Table) | 3289 MB | 50 ms | 2021 ms |
| B (Search) | 0.01 MB | 50 ms | 409 ms |
| C (Block) | 0.26 MB | 49 ms | 409 ms |

B and C essentially tied on 4090—offset array fits entirely in 72MB L2 cache.

### YOLO Object Detection Pipeline

This is where it gets practical. I ran YOLOv8 on 200 variable-size frames to see if the preprocessing gains actually matter in a real ML pipeline.

**T4 (200 frames, variable size → 640×640):**

| Method | Preprocess | Inference | Total |
|--------|------------|-----------|-------|
| CPU (PIL) | 2891 ms | 1493 ms | 4384 ms |
| GPU | 507 ms | 983 ms | 1490 ms |

That's **2.9x faster end-to-end** with identical results (93 detections). The CPU on T4 is slow enough that GPU preprocessing isn't optional—it's necessary.

**RTX 4090 (200 frames):**

| Method | Preprocess | Inference | Total |
|--------|------------|-----------|-------|
| CPU (PIL) | 1530 ms | 347 ms | 1877 ms |
| GPU | 208 ms | 939 ms | 1147 ms |

Still 1.6x faster, but the 4090's CPU is beefy enough that you could get away with PIL if you had to.

The takeaway: on weaker hardware, GPU preprocessing matters more, not less.

## Why C Beats B on Small Cache

Binary search does O(log M) memory accesses per pixel. With 500K images:

- log₂(500000) = 19 lookups per pixel
- Offset array = 3.8 MB (doesn't fit in 4MB L2)
- Random access pattern = cache misses

Block metadata does O(1) lookup per CUDA block:

- One read of `block_to_image` per block
- Sequential memory access within block
- Deterministic, cache-friendly

The 28% kernel speedup on T4 (500K images) matches this analysis—when offset array exceeds L2 cache, binary search pays the penalty.

## When to Use What

| Your Situation | Recommendation |
|----------------|----------------|
| High-end GPU (4090, A100) | B or C, doesn't matter |
| Edge device (Jetson, <4MB L2) | **C (block metadata)** |
| Memory constrained | B (smallest footprint) |
| Need per-block scheduling | **C (only option)** |
| Simple batching | B (easier to implement) |

## The Real Win: Scheduling Flexibility

Beyond performance, block metadata enables per-block scheduling:

```python
block_info = {
    'image_id': 3,
    'priority': HIGH,      # process important images first  
    'kernel_type': BLUR,   # different operations per block
    'stream_id': 2,        # multi-stream scheduling
}
```

If you just need image IDs, use B. If you need scheduling flexibility, C is the only option.

## Where This Actually Matters

This probably isn't useful if you're running inference on an A100 in the cloud. It's for the edge cases (literally):

- **Jetson and similar edge devices**: 4MB L2 cache, 102 GB/s memory bandwidth. You can't just throw more hardware at it.
- **Satellite/drone imagery**: Variable-size tiles coming off sensors, real-time processing requirements, power budget measured in watts.
- **Multi-tenant GPU scenarios**: When you're sharing cache with other workloads, deterministic O(1) access beats variable O(log n).
- **Gigapixel medical/satellite images**: A 100K×100K image would need a 40GB lookup table. Block metadata is the only practical option.

## Limitations

A few caveats worth mentioning:

- This is all Numba CUDA, not raw CUDA C. A proper C implementation would be faster for both approaches, though the relative difference should hold.
- D2H transfer time dominates the total runtime in these benchmarks. In a fused pipeline where you don't copy back to host, the kernel speedup would matter more.
- On high-end GPUs with PyTorch available, just use `F.interpolate`. It's faster and easier. This approach is for when you're on edge hardware without PyTorch or need custom kernels.

## Running the Benchmarks

```bash
# Install dependencies
pip install numba numpy pillow torch ultralytics

# Synthetic benchmark
python triple_baseline_benchmark.py --images 10000

# Larger scale (T4/edge focus)
python triple_baseline_benchmark.py --images 100000 --tiny
python triple_baseline_benchmark.py --images 500000 --tiny

# Video frame benchmark (requires ffmpeg)
python video_frame_extract.py --download --extract --prepare
python video_benchmark.py

# YOLO pipeline benchmark
python edge_benchmark.py
```

## Code

The benchmark is Python/Numba CUDA. Setup uses `@njit` for speed.

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

## Hardware Tested

| GPU | L2 Cache | Memory BW | What I found |
|-----|----------|-----------|--------------|
| RTX 4090 | 72 MB | 1 TB/s | B ≈ C. Everything fits in cache. |
| T4 | 4 MB | 320 GB/s | C wins by 22-28%. Cache pressure is real. |
| Jetson Orin Nano | 4 MB | 102 GB/s | Arriving Expecting bigger gap. |

The T4 results were the surprise. I expected some improvement on smaller cache, but 28% on synthetic and 22% on real video data is more than I thought I'd see.

Jetson has the same L2 size as T4 but 3x less memory bandwidth. If binary search is already hurting on T4, it should hurt more on Jetson.

## Files

| File | Purpose |
|------|---------|
| `triple_baseline_benchmark.py` | Synthetic benchmark (A vs B vs C) |
| `video_frame_extract.py` | Extract frames from video via ffmpeg |
| `video_benchmark.py` | Benchmark on real video frames |
| `edge_benchmark.py` | CPU vs GPU preprocessing comparison |
| `yolo_benchmark.py` | YOLOv8 object detection pipeline |

---

Questions or benchmarks on other hardware welcome.
