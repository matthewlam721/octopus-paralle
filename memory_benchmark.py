"""
Memory-Aware Benchmark: Hybrid vs Grid-Stride-Fair
===================================================
Compares TOTAL cost including:
1. Setup time (pre-compute arrays)
2. GPU Memory usage
3. Kernel execution time
4. Total time = Setup + Kernel

Key insight: Grid-Stride-Fair needs O(total_pixels) lookup table,
while Hybrid only needs O(num_blocks) small arrays.

Author: Matthew
Date: January 2026
"""

from numba import cuda
import numpy as np
import time
from pathlib import Path
from PIL import Image
import math

# ============================================
# IMAGE LOADING
# ============================================

def load_images_2d(data_dir, max_images=None):
    """Load images keeping 2D structure info."""
    data_path = Path(data_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(data_path.glob(ext))
    
    image_files = sorted(image_files)[:max_images] if max_images else sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images...")
    
    all_pixels = []
    widths = []
    heights = []
    offsets = []
    sizes = []
    
    current_offset = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert('L')
            w, h = img.size
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            
            all_pixels.append(pixels)
            widths.append(w)
            heights.append(h)
            offsets.append(current_offset)
            sizes.append(w * h)
            current_offset += w * h
    
    images_flat = np.concatenate(all_pixels)
    
    return (images_flat, 
            np.array(offsets, dtype=np.int64),
            np.array(sizes, dtype=np.int64),
            np.array(widths, dtype=np.int32),
            np.array(heights, dtype=np.int32))


def add_large_synthetic_2d(images_flat, offsets, sizes, widths, heights, large_size=(2048, 2048)):
    """Add large synthetic image."""
    w, h = large_size
    large_pixels = np.random.rand(w * h).astype(np.float32)
    
    new_flat = np.concatenate([images_flat, large_pixels])
    new_offsets = np.concatenate([offsets, [len(images_flat)]])
    new_sizes = np.concatenate([sizes, [w * h]])
    new_widths = np.concatenate([widths, [w]])
    new_heights = np.concatenate([heights, [h]])
    
    return new_flat, new_offsets, new_sizes, new_widths, new_heights


# ============================================
# SETUP FUNCTIONS (TIMED)
# ============================================

def setup_grid_stride_fair(offsets, sizes, total_pixels):
    """
    Build pixel_to_image array for O(1) lookup.
    This is O(total_pixels) time and space!
    """
    pixel_to_image = np.zeros(total_pixels, dtype=np.int32)
    
    for img_id in range(len(sizes)):
        start = offsets[img_id]
        end = start + sizes[img_id]
        pixel_to_image[start:end] = img_id
    
    return pixel_to_image


def setup_hybrid(sizes, threshold=65536):
    """
    Build block assignment arrays.
    This is O(num_images) time and O(num_blocks) space!
    """
    block_to_image = []
    block_start = []
    block_end = []
    
    for img_id, size in enumerate(sizes):
        if size <= threshold:
            block_to_image.append(img_id)
            block_start.append(0)
            block_end.append(size)
        else:
            num_blocks = math.ceil(size / threshold)
            pixels_per_block = math.ceil(size / num_blocks)
            
            for b in range(num_blocks):
                block_to_image.append(img_id)
                start = b * pixels_per_block
                end = min((b + 1) * pixels_per_block, size)
                block_start.append(start)
                block_end.append(end)
    
    return (np.array(block_to_image, dtype=np.int32),
            np.array(block_start, dtype=np.int64),
            np.array(block_end, dtype=np.int64))


# ============================================
# KERNELS
# ============================================

@cuda.jit
def grid_stride_fair_blur_kernel(images_flat, offsets, widths, heights,
                                  pixel_to_image, output):
    """Grid-Stride with O(1) pixel_to_image lookup (FAIR baseline)."""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        img_id = pixel_to_image[pixel_idx]
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[pixel_idx] = total / 9.0


@cuda.jit
def hybrid_blur_kernel(images_flat, offsets, widths, heights,
                       block_to_image, block_start, block_end, output):
    """Hybrid: Adaptive block assignment."""
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_image.shape[0]:
        return
    
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images_flat[global_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[global_idx] = total / 9.0


# ============================================
# MEMORY-AWARE BENCHMARK
# ============================================

def run_memory_benchmark(images_flat, offsets, sizes, widths, heights, name,
                         threshold=65536, warmup=3, runs=20):
    """
    Benchmark comparing TOTAL cost:
    - Setup time (CPU)
    - Memory usage
    - Kernel time (GPU)
    - Total = Setup + Kernel
    """
    
    print(f"\n{'='*70}")
    print(f"MEMORY-AWARE BENCHMARK: {name}")
    print(f"{'='*70}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    imbalance = max(sizes) / (total_pixels / num_images)
    
    print(f"\nDataset:")
    print(f"  Images: {num_images:,}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Imbalance: {imbalance:.1f}x")
    
    threads_per_block = 256
    grid_blocks = 256
    
    # Common GPU arrays
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    results = {}
    
    # ========================================
    # METHOD 1: Grid-Stride Fair
    # ========================================
    print(f"\n[Grid-Stride Fair]")
    
    # Setup time (average of multiple runs)
    setup_times = []
    for _ in range(5):
        start = time.perf_counter()
        pixel_to_image = setup_grid_stride_fair(offsets, sizes, total_pixels)
        setup_times.append(time.perf_counter() - start)
    
    grid_setup_time = np.mean(setup_times) * 1000
    grid_setup_std = np.std(setup_times) * 1000
    
    # Memory usage
    grid_memory_bytes = pixel_to_image.nbytes
    grid_memory_mb = grid_memory_bytes / (1024 * 1024)
    
    print(f"  Setup time: {grid_setup_time:.2f} ms (±{grid_setup_std:.2f})")
    print(f"  Memory: {grid_memory_mb:.2f} MB ({total_pixels:,} × 4 bytes)")
    
    # Transfer to GPU
    d_pixel_to_image = cuda.to_device(pixel_to_image)
    
    # Warmup
    for _ in range(warmup):
        grid_stride_fair_blur_kernel[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
    cuda.synchronize()
    
    # Kernel time
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        grid_stride_fair_blur_kernel[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
        cuda.synchronize()
        kernel_times.append(time.perf_counter() - start)
    
    grid_kernel_time = np.mean(kernel_times) * 1000
    grid_kernel_std = np.std(kernel_times) * 1000
    grid_total_time = grid_setup_time + grid_kernel_time
    
    print(f"  Kernel time: {grid_kernel_time:.2f} ms (±{grid_kernel_std:.2f})")
    print(f"  TOTAL time: {grid_total_time:.2f} ms")
    
    results['grid_fair'] = {
        'setup_ms': grid_setup_time,
        'memory_mb': grid_memory_mb,
        'memory_bytes': grid_memory_bytes,
        'kernel_ms': grid_kernel_time,
        'total_ms': grid_total_time
    }
    
    # Free GPU memory
    del d_pixel_to_image
    
    # ========================================
    # METHOD 2: Hybrid
    # ========================================
    print(f"\n[Hybrid]")
    
    # Setup time
    setup_times = []
    for _ in range(5):
        start = time.perf_counter()
        block_to_image, block_start, block_end = setup_hybrid(sizes, threshold)
        setup_times.append(time.perf_counter() - start)
    
    hybrid_setup_time = np.mean(setup_times) * 1000
    hybrid_setup_std = np.std(setup_times) * 1000
    
    # Memory usage
    hybrid_memory_bytes = block_to_image.nbytes + block_start.nbytes + block_end.nbytes
    hybrid_memory_mb = hybrid_memory_bytes / (1024 * 1024)
    num_blocks = len(block_to_image)
    
    print(f"  Setup time: {hybrid_setup_time:.3f} ms (±{hybrid_setup_std:.3f})")
    print(f"  Memory: {hybrid_memory_mb:.4f} MB ({num_blocks:,} blocks × 20 bytes)")
    print(f"  Blocks: {num_blocks:,}")
    
    # Transfer to GPU
    d_block_to_image = cuda.to_device(block_to_image)
    d_block_start = cuda.to_device(block_start)
    d_block_end = cuda.to_device(block_end)
    
    # Warmup
    for _ in range(warmup):
        hybrid_blur_kernel[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    # Kernel time
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        hybrid_blur_kernel[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
        cuda.synchronize()
        kernel_times.append(time.perf_counter() - start)
    
    hybrid_kernel_time = np.mean(kernel_times) * 1000
    hybrid_kernel_std = np.std(kernel_times) * 1000
    hybrid_total_time = hybrid_setup_time + hybrid_kernel_time
    
    print(f"  Kernel time: {hybrid_kernel_time:.2f} ms (±{hybrid_kernel_std:.2f})")
    print(f"  TOTAL time: {hybrid_total_time:.2f} ms")
    
    results['hybrid'] = {
        'setup_ms': hybrid_setup_time,
        'memory_mb': hybrid_memory_mb,
        'memory_bytes': hybrid_memory_bytes,
        'kernel_ms': hybrid_kernel_time,
        'total_ms': hybrid_total_time
    }
    
    # ========================================
    # COMPARISON
    # ========================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    setup_speedup = grid_setup_time / hybrid_setup_time
    memory_ratio = grid_memory_bytes / hybrid_memory_bytes
    kernel_ratio = grid_kernel_time / hybrid_kernel_time
    total_speedup = grid_total_time / hybrid_total_time
    
    print(f"\n  {'Metric':<20} {'Grid-Fair':>15} {'Hybrid':>15} {'Ratio':>15}")
    print(f"  {'-'*65}")
    print(f"  {'Setup time':<20} {grid_setup_time:>14.2f}ms {hybrid_setup_time:>14.3f}ms {setup_speedup:>14.1f}x")
    print(f"  {'Memory':<20} {grid_memory_mb:>14.2f}MB {hybrid_memory_mb:>14.4f}MB {memory_ratio:>14.0f}x")
    print(f"  {'Kernel time':<20} {grid_kernel_time:>14.2f}ms {hybrid_kernel_time:>14.2f}ms {kernel_ratio:>14.2f}x")
    print(f"  {'-'*65}")
    print(f"  {'TOTAL time':<20} {grid_total_time:>14.2f}ms {hybrid_total_time:>14.2f}ms {total_speedup:>14.2f}x")
    
    # Winner determination
    print(f"\n  Results:")
    if total_speedup > 1.05:
        print(f"  >>> HYBRID WINS (Total: {total_speedup:.2f}x faster) <<<")
    elif total_speedup > 0.95:
        print(f"  >>> TIE (within 5%) <<<")
    else:
        print(f"  >>> GRID-STRIDE WINS <<<")
    
    print(f"  >>> HYBRID uses {memory_ratio:.0f}x LESS memory <<<")
    print(f"  >>> HYBRID setup is {setup_speedup:.0f}x faster <<<")
    
    return {
        'name': name,
        'num_images': num_images,
        'total_pixels': total_pixels,
        'imbalance': imbalance,
        'results': results,
        'setup_speedup': setup_speedup,
        'memory_ratio': memory_ratio,
        'kernel_ratio': kernel_ratio,
        'total_speedup': total_speedup
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("MEMORY-AWARE BENCHMARK")
    print("Comparing TOTAL cost: Setup + Memory + Kernel")
    print("=" * 70)
    print()
    print("Key insight:")
    print("  Grid-Stride-Fair needs O(total_pixels) lookup table")
    print("  Hybrid only needs O(num_blocks) small arrays")
    print()
    
    all_results = []
    
    # ========================================
    # Test 1: Flickr Pure
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 1: Flickr Pure (500 images)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        r = run_memory_benchmark(*data, "Flickr Pure")
        all_results.append(r)
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 2: Flickr + 4K
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 2: Flickr + 4K (high imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        data = add_large_synthetic_2d(*data, (3840, 2160))
        r = run_memory_benchmark(*data, "Flickr + 4K")
        all_results.append(r)
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 3: Flickr + 8K
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 3: Flickr + 8K (extreme imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        data = add_large_synthetic_2d(*data, (7680, 4320))
        r = run_memory_benchmark(*data, "Flickr + 8K")
        all_results.append(r)
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 4: Large Scale (more images)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 4: Flickr Full (1000 images)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=1000)
        r = run_memory_benchmark(*data, "Flickr 1000")
        all_results.append(r)
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 5: Large Scale + 8K
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 5: Flickr 1000 + 8K")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=1000)
        data = add_large_synthetic_2d(*data, (7680, 4320))
        r = run_memory_benchmark(*data, "Flickr 1000 + 8K")
        all_results.append(r)
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("MEMORY-AWARE BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Test':<20} {'Pixels':>12} {'Setup':>10} {'Memory':>10} {'Kernel':>10} {'TOTAL':>10}")
    print(f"  {'-'*75}")
    
    for r in all_results:
        print(f"  {r['name']:<20} {r['total_pixels']:>11,} {r['setup_speedup']:>9.0f}x {r['memory_ratio']:>9.0f}x {r['kernel_ratio']:>9.2f}x {r['total_speedup']:>9.2f}x")
    
    print(f"\n  (Ratios = Grid-Fair / Hybrid, higher = Hybrid wins)")
    
    # Average stats
    if all_results:
        avg_setup = np.mean([r['setup_speedup'] for r in all_results])
        avg_memory = np.mean([r['memory_ratio'] for r in all_results])
        avg_kernel = np.mean([r['kernel_ratio'] for r in all_results])
        avg_total = np.mean([r['total_speedup'] for r in all_results])
        
        print(f"\n  {'AVERAGE':<20} {'-':>12} {avg_setup:>9.0f}x {avg_memory:>9.0f}x {avg_kernel:>9.2f}x {avg_total:>9.2f}x")
    
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    
    print(f"""
  1. SETUP TIME: Hybrid is {avg_setup:.0f}x faster
     - Grid-Fair: O(total_pixels) to build lookup table
     - Hybrid: O(num_images) to build block arrays
  
  2. MEMORY: Hybrid uses {avg_memory:.0f}x less memory
     - Grid-Fair: {all_results[0]['results']['grid_fair']['memory_mb']:.1f} MB for {all_results[0]['total_pixels']:,} pixels
     - Hybrid: {all_results[0]['results']['hybrid']['memory_mb']:.4f} MB
  
  3. KERNEL: {'Similar' if 0.8 < avg_kernel < 1.2 else 'Different'} performance
     - Both achieve similar throughput once running
  
  4. TOTAL: Hybrid is {avg_total:.2f}x faster overall
     - Setup overhead dominates for Grid-Fair
    """)
    
    # Conclusion
    if avg_total > 1.0:
        print(f"\n  🐙 HYBRID WINS when considering TOTAL cost!")
        print(f"     Setup + Memory savings outweigh any kernel differences")
    else:
        print(f"\n  Grid-Stride-Fair wins on total time")
    
    print("\n" + "=" * 70)
    print("🐙 Memory-aware benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()