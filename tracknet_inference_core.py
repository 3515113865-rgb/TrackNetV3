from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Shuttlecock_Trajectory_Dataset
from test import get_ensemble_weight
from utils.general import HEIGHT, WIDTH, get_model


@dataclass(frozen=True)
class HeatmapDetection:
    frame: int
    visibility: int
    x: float
    y: float
    confidence: float


def load_tracknet(tracknet_file: str, device: torch.device):
    ckpt = torch.load(tracknet_file, map_location=device)
    seq_len = int(ckpt["param_dict"]["seq_len"])
    bg_mode = ckpt["param_dict"]["bg_mode"]
    model = get_model("TrackNet", seq_len, bg_mode).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, seq_len, bg_mode


def load_inpaintnet(inpaintnet_file: str, device: torch.device):
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


def video_metadata(video_file: str) -> tuple[int, float, int, int]:
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


def read_frame_range(video_file: str, start: int, end: int) -> list[np.ndarray]:
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
) -> list[HeatmapDetection]:
    if not frames_bgr:
        return []
    if len(frames_bgr) != len(frame_ids):
        raise ValueError("frames_bgr and frame_ids must have the same length")

    tile_h, tile_w = frames_bgr[0].shape[:2]
    img_scaler = (tile_w / WIDTH, tile_h / HEIGHT)
    rgb_frames = bgr_to_rgb_array(frames_bgr)
    sample_count = min(len(rgb_frames), max(1, background_sample_frames))
    sample_indices = np.linspace(0, len(rgb_frames) - 1, sample_count, dtype=int)
    median = np.median(rgb_frames[sample_indices], axis=0) if bg_mode else None
    dataset = Shuttlecock_Trajectory_Dataset(
        seq_len=seq_len,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode=bg_mode,
        frame_arr=rgb_frames,
        median=median,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    video_len = len(frames_bgr)
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
    weight = get_ensemble_weight(seq_len, eval_mode)
    num_sample = max(video_len - seq_len + 1, 0)
    sample_count = 0
    heatmaps_by_local_frame: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for i, x in tqdm(loader, desc=progress_desc):
            x = x.float().to(device)
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
