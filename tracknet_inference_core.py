from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import Shuttlecock_Trajectory_Dataset
from test import get_ensemble_weight
from utils.general import HEIGHT, WIDTH, get_model


class VideoSegmentReader:
    """Sequential bounded reader that reuses only the segment overlap."""

    def __init__(self, video_file: str, profiler=None, purpose: str = "full_frame"):
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_file}")
        self.profiler = profiler
        self.purpose = purpose
        self.tail: list[np.ndarray] = []
        if profiler:
            profiler.add_count("video_open_count")
            profiler.add_count(f"video_open_{purpose}")

    def segments(self, ranges: list[tuple[int, int]]):
        previous_end = 0
        for index, (start, end) in enumerate(ranges):
            overlap = max(0, previous_end - start) if index else 0
            frames = self.tail[-overlap:] if overlap else []
            needed = end - start - len(frames)
            decode_started = time.perf_counter()
            for _ in range(needed):
                ok, frame = self.cap.read()
                if not ok:
                    break
                frames.append(frame)
                if self.profiler:
                    self.profiler.add_count("decoded_frame_count")
                    self.profiler.add_count("full_frame_source_decode_count")
            if self.profiler:
                self.profiler.add_time("full_frame_video_decode", time.perf_counter() - decode_started)
            self.tail = frames
            previous_end = end
            if self.profiler:
                self.profiler.counts["segment_buffer_peak_frames"] = max(
                    self.profiler.counts.get("segment_buffer_peak_frames", 0), len(frames)
                )
            yield start, end, frames

    def close(self):
        self.cap.release()


class VideoRangeReader:
    """One capture for bounded, non-overlapping recovery ranges."""

    def __init__(self, video_file: str, profiler=None):
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_file}")
        self.profiler = profiler
        if profiler:
            profiler.add_count("video_open_count")
            profiler.add_count("video_open_tile_recovery")

    def read(self, start: int, end: int) -> list[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(start, end):
            ok, frame = self.cap.read()
            if not ok:
                break
            frames.append(frame)
            if self.profiler:
                self.profiler.add_count("decoded_frame_count")
                self.profiler.add_count("tile_source_decode_count")
        return frames

    def close(self):
        self.cap.release()


def estimate_video_background(video_file: str, total_frames: int, max_samples: int, profiler=None):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    if profiler:
        profiler.add_count("video_open_count")
        profiler.add_count("video_open_background")
    sample_count = min(total_frames, max(1, max_samples))
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    samples = []
    try:
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            samples.append(frame[:, :, ::-1])
            if profiler:
                profiler.add_count("decoded_frame_count")
                profiler.add_count("background_source_decode_count")
    finally:
        cap.release()
    if profiler:
        profiler.add_count("background_estimation_count")
        profiler.add_count("background_sampled_frames", len(samples))
    return np.median(samples, axis=0) if samples else None


def estimate_frame_background(frames_bgr: list[np.ndarray], max_samples: int, profiler=None):
    sample_count = min(len(frames_bgr), max(1, max_samples))
    indices = np.linspace(0, len(frames_bgr) - 1, sample_count, dtype=int)
    rgb = np.array([frames_bgr[index][:, :, ::-1] for index in indices])
    if profiler:
        profiler.add_count("background_estimation_count")
        profiler.add_count("background_sampled_frames", sample_count)
    return np.median(rgb, axis=0)


@dataclass(frozen=True)
class HeatmapDetection:
    frame: int
    visibility: int
    x: float
    y: float
    confidence: float


def load_tracknet(tracknet_file: str, device: torch.device, profiler=None):
    if profiler:
        profiler.add_count("tracknet_model_load_count")
    ckpt = torch.load(tracknet_file, map_location=device)
    seq_len = int(ckpt["param_dict"]["seq_len"])
    bg_mode = ckpt["param_dict"]["bg_mode"]
    model = get_model("TrackNet", seq_len, bg_mode).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, seq_len, bg_mode


def load_inpaintnet(inpaintnet_file: str, device: torch.device, profiler=None):
    if profiler:
        profiler.add_count("inpaintnet_model_load_count")
    ckpt = torch.load(inpaintnet_file, map_location=device)
    seq_len = int(ckpt["param_dict"]["seq_len"])
    model = get_model("InpaintNet").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, seq_len


def read_video_frames(video_file: str) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height


def video_metadata(video_file: str, profiler=None) -> tuple[int, float, int, int]:
    if profiler:
        profiler.add_count("video_open_count")
        profiler.add_count("video_open_metadata")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def segment_ranges(total_frames: int, segment_frames: int = 180, overlap: int = 12) -> list[tuple[int, int]]:
    if total_frames <= 0:
        return []
    if segment_frames <= overlap or overlap < 0:
        raise ValueError("segment_frames must be greater than segment_overlap")
    ranges, start = [], 0
    while start < total_frames:
        end = min(total_frames, start + segment_frames)
        ranges.append((start, end))
        if end == total_frames:
            break
        start = end - overlap
    return ranges


def read_frame_range(video_file: str, start: int, end: int, profiler=None, purpose: str = "range", count_open: bool = True) -> list[np.ndarray]:
    if profiler and count_open:
        profiler.add_count("video_open_count")
        profiler.add_count(f"video_open_{purpose}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if profiler:
                profiler.add_count("decoded_frame_count")
    finally:
        cap.release()
    return frames


def bgr_to_rgb_array(frames: Iterable[np.ndarray]) -> np.ndarray:
    return np.array([frame[:, :, ::-1] for frame in frames])


def heatmap_to_detection(
    frame_id: int,
    heatmap: np.ndarray,
    img_scaler: tuple[float, float],
    threshold: float,
    offset: tuple[int, int] = (0, 0),
) -> HeatmapDetection:
    confidence = float(np.max(heatmap)) if heatmap.size else 0.0
    if confidence < threshold:
        return HeatmapDetection(frame_id, 0, 0.0, 0.0, confidence)

    mask = (heatmap >= threshold).astype("uint8") * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return HeatmapDetection(frame_id, 0, 0.0, 0.0, confidence)

    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    x, y, w, h = max(rects, key=lambda rect: rect[2] * rect[3])
    local_x = (x + w / 2.0) * img_scaler[0]
    local_y = (y + h / 2.0) * img_scaler[1]
    return HeatmapDetection(
        frame_id,
        1,
        local_x + offset[0],
        local_y + offset[1],
        confidence,
    )


def run_tracknet_heatmap(
    *,
    frames_bgr: list[np.ndarray],
    frame_ids: list[int],
    model,
    seq_len: int,
    bg_mode: str,
    batch_size: int,
    threshold: float,
    device: torch.device,
    offset: tuple[int, int] = (0, 0),
    eval_mode: str = "weight",
    progress_desc: str = "TrackNet",
    background_sample_frames: int = 64,
    profiler=None,
    profile_prefix: str = "full_frame",
    background_rgb: np.ndarray | None = None,
) -> list[HeatmapDetection]:
    pipeline_started = __import__("time").perf_counter()
    inference_before = profiler.timings.get(f"{profile_prefix}_inference", 0.0) if profiler else 0.0
    background_before = profiler.timings.get("background_estimation", 0.0) if profiler else 0.0
    preparation_before = profiler.timings.get(f"{profile_prefix}_preparation", 0.0) if profiler else 0.0
    if not frames_bgr:
        return []
    if len(frames_bgr) != len(frame_ids):
        raise ValueError("frames_bgr and frame_ids must have the same length")

    tile_h, tile_w = frames_bgr[0].shape[:2]
    img_scaler = (tile_w / WIDTH, tile_h / HEIGHT)
    stage = profiler.stage if profiler else None
    rgb_frames = bgr_to_rgb_array(frames_bgr)
    median = background_rgb
    preprocess_name = f"{profile_prefix}_preprocess"
    with stage(preprocess_name) if stage else _nullcontext():
        prepared = []
        for rgb in rgb_frames:
            if bg_mode == "subtract":
                diff = np.sum(np.absolute(rgb - median), axis=2).astype("uint8")
                resized = np.asarray(Image.fromarray(diff).resize((WIDTH, HEIGHT)))[None, ...]
            elif bg_mode == "subtract_concat":
                diff = np.sum(np.absolute(rgb - median), axis=2).astype("uint8")
                resized_rgb = np.moveaxis(np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT))), -1, 0)
                resized_diff = np.asarray(Image.fromarray(diff).resize((WIDTH, HEIGHT)))[None, ...]
                resized = np.concatenate((resized_rgb, resized_diff), axis=0)
            else:
                resized = np.moveaxis(np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT))), -1, 0)
            prepared.append(resized)
        prepared = np.ascontiguousarray(np.stack(prepared))
        if bg_mode == "concat":
            resized_median = np.moveaxis(np.asarray(Image.fromarray(median.astype("uint8")).resize((WIDTH, HEIGHT))), -1, 0)

    video_len = len(frames_bgr)
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
    weight = get_ensemble_weight(seq_len, eval_mode)
    num_sample = max(video_len - seq_len + 1, 0)
    sample_count = 0
    heatmaps_by_local_frame: dict[int, np.ndarray] = {}

    starts = list(range(num_sample))
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_sample, batch_size), desc=progress_desc):
            batch_starts = starts[batch_start:batch_start + batch_size]
            with stage(f"{profile_prefix}_preparation") if stage else _nullcontext():
                arrays = []
                indices = []
                for start in batch_starts:
                    window = prepared[start:start + seq_len].reshape(-1, HEIGHT, WIDTH)
                    if bg_mode == "concat":
                        window = np.concatenate((resized_median, window), axis=0)
                    arrays.append(window)
                    indices.append([(0, frame) for frame in range(start, start + seq_len)])
                x = torch.from_numpy(np.ascontiguousarray(np.stack(arrays))).float().div_(255.0)
                i = torch.tensor(indices, dtype=torch.int64)
            with stage(f"{profile_prefix}_host_to_device", synchronize_cuda=True) if stage else _nullcontext():
                x = x.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else x.to(device)
            if profiler:
                profiler.observe_inference(model, x)
            with stage(f"{profile_prefix}_inference", synchronize_cuda=True) if stage else _nullcontext():
                y_pred = model(x).detach().cpu()
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            b_size = int(i.shape[0])

            for b in range(b_size):
                if sample_count < buffer_size:
                    y_frame = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    y_frame = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                local_frame = int(i[b][0][1])
                heatmaps_by_local_frame[local_frame] = y_frame.numpy()
                sample_count += 1

                if sample_count == num_sample:
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                    for f in range(1, seq_len):
                        y_tail = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        local_tail = int(i[-1][f][1])
                        heatmaps_by_local_frame[local_tail] = y_tail.numpy()

            y_pred_buffer = y_pred_buffer[-buffer_size:]

    with stage("coordinate_mapping") if stage else _nullcontext():
        detections: list[HeatmapDetection] = []
        for local_frame, global_frame in enumerate(frame_ids):
            heatmap = heatmaps_by_local_frame.get(local_frame)
            if heatmap is None:
                detections.append(HeatmapDetection(global_frame, 0, 0.0, 0.0, 0.0))
            else:
                detections.append(
                    heatmap_to_detection(
                        global_frame,
                        heatmap,
                        img_scaler,
                        threshold,
                        offset=offset,
                    )
                )
    return detections


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
