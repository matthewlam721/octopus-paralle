"""
Web Image Benchmark (Real Data)
================================
Using real Flickr8k images to validate octopus load balancing
for web/CDN image processing scenarios.

Author: Matthew
Date: January 28, 2026
"""

from numba import cuda
import numpy as np
import time
from pathlib import Path
from PIL import Image
from scipy import stats
import os

# ============================================
# IMAGE LOADING
# ============================================

def load_web_images(data_dir, max_images=None, grayscale=True):
    """Load web images from Flickr8k dataset."""
    data_path = Path(data_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(data_path.glob(ext))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images from {data_dir}...")
    
    # First pass: get sizes
    image_dims = []
    total_pixels = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')
            w, h = img.size
            image_dims.append((w, h))
            total_pixels += w * h
    
    # Second pass: load pixel data
    images_flat = np.zeros(total_pixels, dtype=np.float32)
    offsets = np.zeros(len(image_files), dtype=np.int64)
    sizes = np.zeros(len(image_files), dtype=np.int64)
    
    current_offset = 0
    for i, img_path in enumerate(image_files):
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            offsets[i] = current_offset
            sizes[i] = len(pixels)
            images_flat[current_offset:current_offset + len(pixels)] = pixels
            current_offset += len(pixels)
    
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Size range: {min(sizes):,} - {max(sizes):,} pixels")
    
    return images_flat, offsets, sizes, image_dims


def add_synthetic_large_images(images_flat, offsets, sizes, dims, large_configs):
    """
    Add synthetic large images to simulate CDN/social media workloads.
    
    large_configs: list of (width, height) for synthetic images
    """
    new_images = []
    new_sizes = []
    new_dims = []
    
    for w, h in large_configs:
        large_size = w * h
        large_img = np.random.rand(large_size).astype(np.float32)
        new_images.append(large_img)
        new_sizes.append(large_size)
        new_dims.append((w, h))
    
    # Concatenate
    combined_flat = np.concatenate([images_flat] + new_images)
    combined_offsets = list(offsets) + [len(images_flat) + sum(new_sizes[:i]) for i in range(len(new_sizes))]
    combined_sizes = np.concatenate([sizes, new_sizes])
    combined_dims = dims + new_dims
    
    return (np.array(combined_flat), 
            np.array(combined_offsets, dtype=np.int64), 
            np.array(combined_sizes, dtype=np.int64), 
            combined_dims)


# ============================================
# GPU KERNELS
# ============================================

@cuda.jit
def naive_kernel(images_flat, image_offsets, image_sizes, output):
    """Naive: Each thread processes one entire image."""
    tid = cuda.grid(1)
    if tid < image_sizes.shape[0]:
        start = image_offsets[tid]
        num_pixels = image_sizes[tid]
        result = 0.0
        for i in range(num_pixels):
            pixel = images_flat[start + i]
            result += pixel * 0.5 + 0.1
            result = result * 1.001
        output[tid] = result


@cuda.jit
def balanced_kernel(images_flat, work_start, work_end, output):
    """Balanced: Each thread processes equal number of pixels."""
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
# BENCHMARK
# ============================================

def compute_balanced_work(total_pixels, num_threads):
    """Compute balanced work distribution."""
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


def run_benchmark(images_flat, offsets, sizes, image_dims, 
                  num_threads, name, warmup=5, runs=30):
    """Run benchmark with full statistical analysis."""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    max_pixels = max(sizes)
    min_pixels = min(sizes)
    avg_pixels = total_pixels / num_images
    imbalance = max_pixels / avg_pixels
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} total pixels")
    print(f"Size range: {min_pixels:,} - {max_pixels:,} (imbalance: {imbalance:.2f}x)")
    
    # Setup
    work_start, work_end = compute_balanced_work(total_pixels, num_threads)
    
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_images, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    threads_per_block = 256
    blocks_naive = (num_images + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    for _ in range(warmup):
        naive_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        balanced_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark
    naive_times = []
    for _ in range(runs):
        start = time.perf_counter()
        naive_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    balanced_times = []
    for _ in range(runs):
        start = time.perf_counter()
        balanced_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    # Statistics
    naive_avg = np.mean(naive_times) * 1000
    naive_std = np.std(naive_times) * 1000
    naive_ci = stats.t.interval(0.95, len(naive_times)-1,
                                loc=naive_avg,
                                scale=stats.sem(naive_times)*1000)
    
    balanced_avg = np.mean(balanced_times) * 1000
    balanced_std = np.std(balanced_times) * 1000
    balanced_ci = stats.t.interval(0.95, len(balanced_times)-1,
                                   loc=balanced_avg,
                                   scale=stats.sem(balanced_times)*1000)
    
    t_stat, p_value = stats.ttest_ind(naive_times, balanced_times)
    speedup = naive_avg / balanced_avg
    
    # Print results
    print(f"\nResults (n={runs}):")
    print(f"  Naive:    {naive_avg:.3f} ms (±{naive_std:.3f})")
    print(f"            95% CI: [{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]")
    print(f"  Balanced: {balanced_avg:.3f} ms (±{balanced_std:.3f})")
    print(f"            95% CI: [{balanced_ci[0]:.3f}, {balanced_ci[1]:.3f}]")
    print(f"\n  Speedup:  {speedup:.2f}x")
    print(f"  p-value:  {p_value:.2e}")
    
    if p_value < 0.001:
        print(f"  >>> HIGHLY SIGNIFICANT (p < 0.001) <<<")
    elif p_value < 0.05:
        print(f"  >>> SIGNIFICANT (p < 0.05) <<<")
    
    return {
        'name': name,
        'num_images': num_images,
        'total_pixels': total_pixels,
        'imbalance': imbalance,
        'naive_ms': naive_avg,
        'balanced_ms': balanced_avg,
        'speedup': speedup,
        'p_value': p_value
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("WEB IMAGE BENCHMARK")
    print("Octopus Load Balancing - Flickr8k Dataset")
    print("=" * 70)
    print(f"\nGPU: {cuda.gpus}")
    
    WEB_DIR = os.path.expanduser("~/cuda-test/Images")
    
    results = []
    
    # ========================================
    # TEST 1: Pure Flickr images (small batch)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 1: Flickr Images Only (500 images)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=500)
    r = run_benchmark(web_flat, web_off, web_sizes, web_dims,
                      num_threads=len(web_sizes),
                      name="Flickr 500 (Pure)")
    results.append(r)
    
    # ========================================
    # TEST 2: Pure Flickr images (full dataset)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 2: Flickr Images Only (Full ~8000 images)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=None)
    r = run_benchmark(web_flat, web_off, web_sizes, web_dims,
                      num_threads=len(web_sizes),
                      name="Flickr Full (Pure)")
    results.append(r)
    
    # ========================================
    # TEST 3: CDN Scenario - Thumbnails + Full-res
    # Simulate: 1000 thumbnails + 1 original 4K image
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 3: CDN Scenario (1000 Flickr + 1x 4K)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=1000)
    mixed_flat, mixed_off, mixed_sizes, mixed_dims = add_synthetic_large_images(
        web_flat, web_off, web_sizes, web_dims,
        large_configs=[(3840, 2160)]  # 4K
    )
    r = run_benchmark(mixed_flat, mixed_off, mixed_sizes, mixed_dims,
                      num_threads=len(mixed_sizes),
                      name="CDN (1000 + 1x4K)")
    results.append(r)
    
    # ========================================
    # TEST 4: Social Media Scenario
    # Simulate: 500 previews + 5 original high-res uploads
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 4: Social Media (500 Flickr + 5x 4K)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=500)
    mixed_flat, mixed_off, mixed_sizes, mixed_dims = add_synthetic_large_images(
        web_flat, web_off, web_sizes, web_dims,
        large_configs=[(3840, 2160)] * 5  # 5x 4K images
    )
    r = run_benchmark(mixed_flat, mixed_off, mixed_sizes, mixed_dims,
                      num_threads=len(mixed_sizes),
                      name="Social Media (500 + 5x4K)")
    results.append(r)
    
    # ========================================
    # TEST 5: E-commerce Scenario
    # Simulate: 2000 product thumbnails + 10 hero images (8K)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 5: E-commerce (2000 Flickr + 10x 8K)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=2000)
    mixed_flat, mixed_off, mixed_sizes, mixed_dims = add_synthetic_large_images(
        web_flat, web_off, web_sizes, web_dims,
        large_configs=[(7680, 4320)] * 10  # 10x 8K hero images
    )
    r = run_benchmark(mixed_flat, mixed_off, mixed_sizes, mixed_dims,
                      num_threads=len(mixed_sizes),
                      name="E-commerce (2000 + 10x8K)")
    results.append(r)
    
    # ========================================
    # TEST 6: Extreme CDN - Many small + 1 huge
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 6: Extreme CDN (5000 Flickr + 1x 16K)")
    print("█" * 70)
    
    web_flat, web_off, web_sizes, web_dims = load_web_images(WEB_DIR, max_images=5000)
    mixed_flat, mixed_off, mixed_sizes, mixed_dims = add_synthetic_large_images(
        web_flat, web_off, web_sizes, web_dims,
        large_configs=[(15360, 8640)]  # 16K (extreme)
    )
    r = run_benchmark(mixed_flat, mixed_off, mixed_sizes, mixed_dims,
                      num_threads=len(mixed_sizes),
                      name="Extreme CDN (5000 + 1x16K)")
    results.append(r)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("WEB IMAGE BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Scenario':<30} {'Images':>8} {'Imbalance':>10} {'Speedup':>10} {'p-value':>12} {'Status':>8}")
    print("-" * 82)
    
    for r in results:
        status = "✓ WIN" if r['speedup'] > 1.05 else ("~ TIE" if r['speedup'] > 0.95 else "✗ LOSE")
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.01 else f"{r['p_value']:.4f}"
        print(f"{r['name']:<30} {r['num_images']:>8} {r['imbalance']:>9.2f}x {r['speedup']:>9.2f}x {p_str:>12} {status:>8}")
    
    print("\n" + "=" * 70)
    
    wins = sum(1 for r in results if r['speedup'] > 1.05)
    significant = sum(1 for r in results if r['p_value'] < 0.001)
    
    print(f"\nResults: {wins}/{len(results)} show improvement")
    print(f"Statistically significant (p < 0.001): {significant}/{len(results)}")
    
    if wins > 0:
        best = max(results, key=lambda x: x['speedup'])
        print(f"Best speedup: {best['speedup']:.2f}x on '{best['name']}'")
    
    print("\n🐙 Web image benchmark complete!")
    
    return results


if __name__ == "__main__":
    results = main()