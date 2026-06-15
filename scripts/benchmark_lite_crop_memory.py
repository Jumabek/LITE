#!/usr/bin/env python3
"""Benchmark LITE feature-map crop memory access on CUDA.

The benchmark mirrors LITE's extraction path:
    crop = feature_map[:, y1:y2, x1:x2]
    descriptor = normalize(mean(crop, dim=(1, 2)))

It separates Python/view creation from the CUDA reduction that actually reads
the non-contiguous crop regions from GPU memory.
"""

import argparse
import statistics
import time

import torch


def make_boxes(num_boxes, h_map, w_map, crop_h, crop_w, mode, device):
    if crop_h > h_map or crop_w > w_map:
        raise ValueError("crop size must fit within the feature-map dimensions")

    if mode == "fixed":
        y1 = torch.full((num_boxes,), (h_map - crop_h) // 2, device=device, dtype=torch.long)
        x1 = torch.full((num_boxes,), (w_map - crop_w) // 2, device=device, dtype=torch.long)
    else:
        y1 = torch.randint(0, h_map - crop_h + 1, (num_boxes,), device=device)
        x1 = torch.randint(0, w_map - crop_w + 1, (num_boxes,), device=device)

    y2 = y1 + crop_h
    x2 = x1 + crop_w
    return torch.stack((y1, x1, y2, x2), dim=1).cpu().tolist()


def lite_reduce(feature_map, boxes):
    features = []
    for y1, x1, y2, x2 in boxes:
        crop = feature_map[:, y1:y2, x1:x2]
        feature_mean = torch.mean(crop, dim=(1, 2))
        norm = feature_mean.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        features.append(feature_mean / norm)
    return torch.stack(features, dim=0)


def crop_views_only(feature_map, boxes):
    views = []
    for y1, x1, y2, x2 in boxes:
        views.append(feature_map[:, y1:y2, x1:x2])
    return views


def time_cuda(fn, warmup, repeats):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    # Keep the output live so the timed work cannot be optimized away.
    if isinstance(out, torch.Tensor):
        checksum = float(out.sum().detach().cpu())
    else:
        checksum = float(sum(v.numel() for v in out))
    return times_ms, checksum


def time_cpu(fn, repeats):
    times_ms = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    checksum = float(sum(v.numel() for v in result))
    return times_ms, checksum


def summarize(times_ms):
    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=576)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--crop-height", type=int, default=22)
    parser.add_argument("--crop-width", type=int, default=22)
    parser.add_argument("--boxes", type=int, nargs="+", default=[1, 10, 25, 50, 100, 200])
    parser.add_argument("--mode", choices=["random", "fixed"], default="random")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    torch.manual_seed(7)
    torch.cuda.set_device(device)
    feature_map = torch.randn(
        args.channels, args.height, args.width, device=device, dtype=torch.float32
    )

    print(f"device,{torch.cuda.get_device_name(device)}")
    print(f"feature_map,C={args.channels},H={args.height},W={args.width}")
    print(f"crop,H={args.crop_height},W={args.crop_width},mode={args.mode}")
    print(
        "boxes,view_create_ms_mean,view_create_ms_median,"
        "reduce_ms_mean,reduce_ms_median,reduce_ms_stdev,"
        "bytes_read_MB,effective_GBps,checksum"
    )

    for num_boxes in args.boxes:
        boxes = make_boxes(
            num_boxes,
            args.height,
            args.width,
            args.crop_height,
            args.crop_width,
            args.mode,
            device,
        )
        view_times, _ = time_cpu(lambda: crop_views_only(feature_map, boxes), args.repeats)
        reduce_times, checksum = time_cuda(
            lambda: lite_reduce(feature_map, boxes), args.warmup, args.repeats
        )
        view_stats = summarize(view_times)
        reduce_stats = summarize(reduce_times)
        bytes_read = num_boxes * args.channels * args.crop_height * args.crop_width * 4
        gbps = (bytes_read / 1e9) / (reduce_stats["median"] / 1000.0)
        print(
            f"{num_boxes},"
            f"{view_stats['mean']:.4f},{view_stats['median']:.4f},"
            f"{reduce_stats['mean']:.4f},{reduce_stats['median']:.4f},{reduce_stats['stdev']:.4f},"
            f"{bytes_read / 1e6:.3f},{gbps:.2f},{checksum:.6f}"
        )


if __name__ == "__main__":
    main()
