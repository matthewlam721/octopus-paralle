"""
Image Processing Benchmark
===========================

Testing pre-balanced workload distribution on variable-size image processing.

Scenario: Batch of images with very different sizes
- Traditional: Each thread processes one image (imbalanced)
- Balanced: Distribute total pixels evenly across threads

Author: Matthew
Date: January 27, 2025
"""

from numba import cuda
import numpy as np
import time
import math

# ============================================
# IMAGE PROCESSING KERNELS
# ============================================

@cuda.jit
def naive_image_kernel(images_flat, image_offsets, image_sizes, output):
    """
    Naive: Each thread processes one entire image.
    
    Problem: Thread with 8192x8192 image takes 256x longer than 512x512.
    """
    tid = cuda.grid(1)
    
    if tid < image_sizes.shape[0]:
        start = image_offsets[tid]
        num_pixels = image_sizes[tid]
        
        # Simulate image processing (e.g., filter, transform)
        result = 0.0
        for i in range(num_pixels):
            pixel = images_flat[start + i]
            # Simulate computation: edge detection-like operation
            result += pixel * 0.5 + 0.1
            result = result * 1.001  # Prevent optimization
        
        output[tid] = result


@cuda.jit
def balanced_image_kernel(images_flat, work_start, work_end, output):
    """
    Balanced: Each thread processes equal number of pixels.
    
    Threads work on contiguous ranges in flattened pixel array.
    """
    tid = cuda.grid(1)
    
    if tid < work_start.shape[0]:
        start = work_start[tid]
        end = work_end[tid]
        
        result = 0.0
        for i in range(start, end):
            pixel = images_flat[i]
            result += pixel * 0.5 + 0.1
            result = result * 1.001
        
        output[tid] = result


# ============================================
# SETUP FUNCTIONS
# ============================================

def create_imbalanced_images(size_configs):
    """
    Create batch of images with different sizes.
    
    size_configs: list of (width, height) tuples
    Returns: flattened array, offsets, sizes
    """
    total_pixels = sum(w * h for w, h in size_configs)
    
    # Create flattened pixel array
    images_flat = np.random.rand(total_pixels).astype(np.float32)
    
    # Calculate offsets and sizes
    offsets = np.zeros(len(size_configs), dtype=np.int64)
    sizes = np.zeros(len(size_configs), dtype=np.int64)
    
    current_offset = 0
    for i, (w, h) in enumerate(size_configs):
        offsets[i] = current_offset
        sizes[i] = w * h
        current_offset += w * h
    
    return images_flat, offsets, sizes


def compute_balanced_work(total_pixels, num_threads):
    """
    Compute balanced work distribution.
    Returns start/end indices for each thread.
    """
    pixels_per_thread = total_pixels // num_threads
    remainder = total_pixels % num_threads
    
    work_start = np.zeros(num_threads, dtype=np.int64)
    work_end = np.zeros(num_threads, dtype=np.int64)
    
    current = 0
    for tid in range(num_threads):
        work_start[tid] = current
        thread_work = pixels_per_thread + (1 if tid < remainder else 0)
        current += thread_work
        work_end[tid] = current
    
    return work_start, work_end


# ============================================
# BENCHMARK
# ============================================

def run_image_benchmark(size_configs, num_threads, name, warmup=3, runs=10):
    """Run comparison benchmark for image processing."""
    
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {name}")
    print("=" * 60)
    
    # Setup
    num_images = len(size_configs)
    images_flat, offsets, sizes = create_imbalanced_images(size_configs)
    total_pixels = len(images_flat)
    max_image_pixels = max(sizes)
    
    print(f"\nConfiguration:")
    print(f"  Images: {num_images}")
    print(f"  Sizes (pixels): {[int(s) for s in sizes]}")
    print(f"  Sizes (dimensions): {size_configs}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Max single image: {max_image_pixels:,} pixels")
    print(f"  Threads: {num_threads}")
    
    # Imbalance analysis
    avg_pixels = total_pixels / num_images
    imbalance = max_image_pixels / avg_pixels
    print(f"  Imbalance ratio: {imbalance:.2f}x")
    
    # Theoretical speedup
    theoretical = max_image_pixels / (total_pixels / num_threads)
    print(f"  Theoretical max speedup: {theoretical:.2f}x")
    
    # Balanced work distribution
    work_start, work_end = compute_balanced_work(total_pixels, num_threads)
    
    print(f"\nWork distribution:")
    print(f"  Naive (per image): {[int(s) for s in sizes]}")
    balanced_work = [work_end[i] - work_start[i] for i in range(num_threads)]
    print(f"  Balanced (per thread): {balanced_work}")
    
    # Transfer to GPU
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_images, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    # Kernel config
    threads_per_block = 256
    blocks_naive = (num_images + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    print(f"\nWarmup ({warmup} runs)...")
    for _ in range(warmup):
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark naive
    print(f"Benchmarking NAIVE ({runs} runs)...")
    naive_times = []
    for _ in range(runs):
        start = time.perf_counter()
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    # Benchmark balanced
    print(f"Benchmarking BALANCED ({runs} runs)...")
    balanced_times = []
    for _ in range(runs):
        start = time.perf_counter()
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    # Verify correctness
    out_naive = d_output_naive.copy_to_host()
    out_balanced = d_output_balanced.copy_to_host()
    
    # Results
    naive_avg = np.mean(naive_times) * 1000
    naive_std = np.std(naive_times) * 1000
    balanced_avg = np.mean(balanced_times) * 1000
    balanced_std = np.std(balanced_times) * 1000
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    print(f"\nTiming:")
    print(f"  Naive:    {naive_avg:.3f} ms (±{naive_std:.3f})")
    print(f"  Balanced: {balanced_avg:.3f} ms (±{balanced_std:.3f})")
    
    speedup = naive_avg / balanced_avg
    
    if speedup > 1:
        print(f"\n  >>> SPEEDUP: {speedup:.2f}x <<<")
        print(f"  >>> Time saved: {(1-balanced_avg/naive_avg)*100:.1f}% <<<")
        print(f"  >>> Achieved {speedup/theoretical*100:.1f}% of theoretical max <<<")
    else:
        print(f"\n  Balanced was {1/speedup:.2f}x slower")
    
    print("=" * 60)
    
    return {
        'name': name,
        'naive_ms': naive_avg,
        'balanced_ms': balanced_avg,
        'speedup': speedup,
        'theoretical': theoretical,
        'imbalance': imbalance,
        'total_pixels': total_pixels
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("IMAGE PROCESSING BENCHMARK")
    print("Octopus-Inspired Load Balancing")
    print("=" * 60)
    print(f"\nGPU: {cuda.gpus}")
    
    results = []
    
    # ----------------------------------------
    # TEST 1: Typical web images (mixed sizes)
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 1: Typical web images (mixed sizes)")
    print("█" * 60)
    sizes_1 = [
        (256, 256),    # Thumbnail
        (1920, 1080),  # Full HD
        (512, 512),    # Medium
        (4096, 2160),  # 4K
    ]
    r = run_image_benchmark(sizes_1, num_threads=4, name="Web Images")
    results.append(r)
    
    # ----------------------------------------
    # TEST 2: Extreme - thumbnails + 8K
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 2: Thumbnails + 8K image")
    print("█" * 60)
    sizes_2 = [
        (64, 64),      # Tiny thumbnail
        (64, 64),      # Tiny thumbnail
        (64, 64),      # Tiny thumbnail
        (7680, 4320),  # 8K image
    ]
    r = run_image_benchmark(sizes_2, num_threads=4, name="Thumbnails + 8K")
    results.append(r)
    
    # ----------------------------------------
    # TEST 3: Medical imaging scenario
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 3: Medical imaging (CT slices + full scan)")
    print("█" * 60)
    sizes_3 = [
        (512, 512),    # CT slice
        (512, 512),    # CT slice
        (512, 512),    # CT slice
        (512, 512),    # CT slice
        (512, 512),    # CT slice
        (4096, 4096),  # Full resolution scan
    ]
    r = run_image_benchmark(sizes_3, num_threads=6, name="Medical Imaging")
    results.append(r)
    
    # ----------------------------------------
    # TEST 4: Satellite imagery
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 4: Satellite imagery (tiles + overview)")
    print("█" * 60)
    sizes_4 = [
        (256, 256),    # Tile
        (256, 256),    # Tile
        (256, 256),    # Tile
        (256, 256),    # Tile
        (256, 256),    # Tile
        (256, 256),    # Tile
        (256, 256),    # Tile
        (10000, 10000), # Full satellite image
    ]
    r = run_image_benchmark(sizes_4, num_threads=8, name="Satellite Imagery")
    results.append(r)
    
    # ----------------------------------------
    # TEST 5: Large scale batch
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 5: Large scale - video frames + keyframe")
    print("█" * 60)
    sizes_5 = (
        [(640, 360)] * 29 +  # 29 low-res frames
        [(3840, 2160)]       # 1 4K keyframe
    )
    r = run_image_benchmark(sizes_5, num_threads=30, name="Video Frames")
    results.append(r)
    
    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Test':<20} {'Pixels':>12} {'Imbalance':>10} {'Theory':>10} {'Actual':>10} {'Status':>8}")
    print("-" * 75)
    
    for r in results:
        status = "✓ WIN" if r['speedup'] > 1.05 else ("~ TIE" if r['speedup'] > 0.95 else "✗ LOSE")
        print(f"{r['name']:<20} {r['total_pixels']:>12,} {r['imbalance']:>9.1f}x {r['theoretical']:>9.2f}x {r['speedup']:>9.2f}x {status:>8}")
    
    print("\n" + "=" * 60)
    
    wins = sum(1 for r in results if r['speedup'] > 1.05)
    print(f"\nBalanced approach wins: {wins}/{len(results)} tests")
    
    if wins > 0:
        best = max(results, key=lambda x: x['speedup'])
        print(f"Best speedup: {best['speedup']:.2f}x on '{best['name']}'")
        print(f"Best time saved: {(1-best['balanced_ms']/best['naive_ms'])*100:.1f}%")
    
    print("\n🐙 Image processing benchmark complete!")
    
    return results


if __name__ == "__main__":
    results = main()
