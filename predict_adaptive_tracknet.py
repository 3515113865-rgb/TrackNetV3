from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Shuttlecock_Trajectory_Dataset
from test import generate_inpaint_mask, get_ensemble_weight
from tracknet_inference_core import (
    HeatmapDetection,
    VideoSegmentReader,
    VideoRangeReader,
    estimate_frame_background,
    load_inpaintnet,
    load_tracknet,
    read_frame_range,
    segment_ranges,
    video_metadata,
    run_tracknet_heatmap,
)
from utils.general import COOR_TH
from tracknet_profiler import TrackNetProfiler


@dataclass(frozen=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive TrackNetV3 inference")
    parser.add_argument("--video_file", required=True)
    parser.add_argument("--tracknet_file", required=True)
    parser.add_argument("--inpaintnet_file", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tiling_mode", choices=["auto", "always", "off"], default="auto")
    parser.add_argument("--max_tiles", type=int, default=4)
    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--candidate_threshold", type=float, default=0.30)
    parser.add_argument("--strong_threshold", type=float, default=0.50)
    parser.add_argument("--max_inpaint_gap", type=int, default=3)
    parser.add_argument("--max_deviation", type=float, default=35.0)
    parser.add_argument("--max_step", type=float, default=120.0)
    parser.add_argument("--corridor_margin", type=float, default=160.0)
    parser.add_argument("--segment_frames", type=int, default=180)
    parser.add_argument("--segment_overlap", type=int, default=12)
    parser.add_argument("--background_sample_frames", type=int, default=64)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def generate_adaptive_tiles(
    frame_width: int,
    frame_height: int,
    max_tiles: int = 4,
    target_aspect_ratio: float = 16 / 9,
    overlap_ratio: float = 0.25,
) -> list[Tile]:
    aspect = frame_width / frame_height
    max_tiles = max(1, max_tiles)

    def axis_windows(total: int, window: int) -> list[int]:
        if window >= total:
            return [0]
        count = max_tiles
        step = max(1, int(window * (1 - overlap_ratio)))
        starts = [0]
        while starts[-1] + window < total and len(starts) < count:
            starts.append(min(total - window, starts[-1] + step))
            if starts[-1] == total - window:
                break
        return sorted(set(starts))

    tiles: list[Tile] = []
    if aspect < 0.85:
        tile_w = frame_width
        tile_h = min(frame_height, int(round(tile_w / target_aspect_ratio)))
        starts = axis_windows(frame_height, tile_h)
        for y in starts[:max_tiles]:
            tiles.append(Tile(len(tiles), 0, y, frame_width, y + tile_h))
    elif aspect > target_aspect_ratio * 1.25:
        tile_h = frame_height
        tile_w = min(frame_width, int(round(tile_h * target_aspect_ratio)))
        starts = axis_windows(frame_width, tile_w)
        for x in starts[:max_tiles]:
            tiles.append(Tile(len(tiles), x, 0, x + tile_w, frame_height))
    elif abs(aspect - target_aspect_ratio) < 0.15:
        tile_h = max(1, frame_height // 2)
        tile_w = min(frame_width, int(round(tile_h * target_aspect_ratio)))
        xs = axis_windows(frame_width, tile_w)[:2]
        if len(xs) == 1:
            xs = [0, max(0, frame_width - tile_w)]
        ys = [0, max(0, frame_height - tile_h)]
        for y in ys:
            for x in xs:
                if len(tiles) < max_tiles:
                    tiles.append(Tile(len(tiles), x, y, x + tile_w, y + tile_h))
    else:
        tile_w = min(frame_width, int(round(frame_height * target_aspect_ratio * 0.75)))
        tile_h = min(frame_height, int(round(tile_w / target_aspect_ratio)))
        xs = [0, max(0, frame_width - tile_w)]
        ys = [0, max(0, frame_height - tile_h)]
        for y in ys:
            for x in xs:
                if len(tiles) < max_tiles:
                    tiles.append(Tile(len(tiles), x, y, x + tile_w, y + tile_h))

    return tiles[:max_tiles]


def point(row) -> tuple[float, float]:
    return float(row["X"]), float(row["Y"])


def find_missing_intervals(full_df: pd.DataFrame) -> list[dict]:
    intervals: list[dict] = []
    visible = full_df["Visibility"].astype(int).tolist()
    frames = full_df["Frame"].astype(int).tolist()
    position = 0
    while position < len(frames):
        if visible[position] == 1:
            position += 1
            continue
        start_pos = position
        while position < len(frames) and visible[position] == 0:
            position += 1
        end_pos = position - 1
        prev_pos = start_pos - 1 if start_pos > 0 and visible[start_pos - 1] == 1 else None
        next_pos = position if position < len(frames) and visible[position] == 1 else None
        prev_row = full_df.iloc[prev_pos] if prev_pos is not None else None
        next_row = full_df.iloc[next_pos] if next_pos is not None else None
        intervals.append(
            {
                "id": len(intervals),
                "start_frame": frames[start_pos],
                "end_frame": frames[end_pos],
                "length": end_pos - start_pos + 1,
                "previous_visible_frame": None if prev_row is None else int(prev_row["Frame"]),
                "next_visible_frame": None if next_row is None else int(next_row["Frame"]),
                "previous_point": None if prev_row is None else [float(prev_row["X"]), float(prev_row["Y"])],
                "next_point": None if next_row is None else [float(next_row["X"]), float(next_row["Y"])],
                "trigger_reason": "tile_gap_len_ge_2" if end_pos - start_pos + 1 >= 2 else "short_gap_for_inpaint",
            }
        )
    return intervals


def merge_supplement_ranges(intervals: list[dict], total_frames: int, context: int, mode: str) -> list[dict]:
    ranges: list[dict] = []
    if mode == "off":
        return ranges
    if mode == "always":
        return [{"start": 0, "end": total_frames - 1, "interval_ids": [item["id"] for item in intervals]}]
    for item in intervals:
        ranges.append(
            {
                "start": max(0, int(item["start_frame"]) - context),
                "end": min(total_frames - 1, int(item["end_frame"]) + context),
                "interval_ids": [int(item["id"])],
            }
        )
    if not ranges:
        return []
    ranges.sort(key=lambda item: item["start"])
    merged = [ranges[0]]
    for item in ranges[1:]:
        last = merged[-1]
        if item["start"] <= last["end"] + 1:
            last["end"] = max(last["end"], item["end"])
            last["interval_ids"].extend(item["interval_ids"])
        else:
            merged.append(item)
    return merged


def rect_intersects_corridor(tile: Tile, prev_point, next_point, margin: float) -> bool:
    if prev_point is None or next_point is None:
        return True
    x1 = min(prev_point[0], next_point[0]) - margin
    x2 = max(prev_point[0], next_point[0]) + margin
    y1 = min(prev_point[1], next_point[1]) - margin
    y2 = max(prev_point[1], next_point[1]) + margin
    return not (tile.x2 < x1 or tile.x1 > x2 or tile.y2 < y1 or tile.y1 > y2)


def choose_tiles_for_interval(tiles: list[Tile], interval: dict, width: int, height: int, margin: float) -> list[Tile]:
    prev_point = interval.get("previous_point")
    next_point = interval.get("next_point")
    adaptive_margin = max(margin, min(width, height) * 0.15)
    chosen = [
        tile for tile in tiles
        if rect_intersects_corridor(tile, prev_point, next_point, adaptive_margin)
    ]
    return chosen or tiles[:1]


def nearest_interval_id(frame: int, intervals: list[dict], range_ids: list[int]) -> int:
    candidates = [intervals[i] for i in range_ids] if range_ids else intervals
    for item in candidates:
        if int(item["start_frame"]) <= frame <= int(item["end_frame"]):
            return int(item["id"])
    return int(candidates[0]["id"]) if candidates else -1


def dedupe_candidates(candidates: pd.DataFrame, radius: float = 25.0) -> pd.DataFrame:
    rows: list[dict] = []
    for frame, group in candidates.groupby("Frame"):
        remaining = group.sort_values("Confidence", ascending=False).to_dict("records")
        while remaining:
            seed = remaining.pop(0)
            merged = [seed]
            keep: list[dict] = []
            for row in remaining:
                if math.dist((seed["X"], seed["Y"]), (row["X"], row["Y"])) <= radius:
                    merged.append(row)
                else:
                    keep.append(row)
            remaining = keep
            best = max(merged, key=lambda row: row["Confidence"])
            best["merged_candidate_count"] = len(merged)
            best["merged_tile_indices"] = ",".join(str(int(row["TileIndex"])) for row in merged)
            rows.append(best)
    return pd.DataFrame(rows)


def scaled_max_step(max_step: float, width: int, height: int, fps: float) -> float:
    base_diag = math.sqrt(720 ** 2 + 1280 ** 2)
    diag = math.sqrt(width ** 2 + height ** 2)
    fps_scale = 30.0 / max(fps, 1.0)
    return max_step * (diag / base_diag) * fps_scale


def temporal_accept(
    candidate: dict,
    last_points: list[tuple[int, float, float]],
    interval: dict | None,
    max_step_px: float,
    strong_threshold: float,
) -> tuple[bool, str, str]:
    conf = float(candidate["Confidence"])
    if not last_points:
        return (conf >= strong_threshold, "validated_strong_start", "low_confidence_start")
    prev_frame, prev_x, prev_y = last_points[-1]
    frame_gap = max(1, int(candidate["Frame"]) - prev_frame)
    step = math.dist((prev_x, prev_y), (candidate["X"], candidate["Y"]))
    if step > max_step_px * frame_gap * 1.5:
        return False, "", "temporal_step_too_large"
    if conf >= strong_threshold:
        return True, "validated_strong", ""
    if len(last_points) >= 2:
        p0 = last_points[-2]
        predicted = (prev_x + (prev_x - p0[1]), prev_y + (prev_y - p0[2]))
        if math.dist(predicted, (candidate["X"], candidate["Y"])) <= max_step_px:
            return True, "validated_weak_predicted", ""
    if interval and interval.get("previous_point") and interval.get("next_point"):
        prev_point = interval["previous_point"]
        next_point = interval["next_point"]
        margin = max_step_px * 1.3
        if rect_intersects_corridor(
            Tile(-1, int(candidate["X"]), int(candidate["Y"]), int(candidate["X"]), int(candidate["Y"])),
            prev_point,
            next_point,
            margin,
        ):
            return True, "validated_weak_corridor", ""
    return False, "", "weak_temporal_rejected"


def fuse_results(
    full_df: pd.DataFrame,
    candidates: pd.DataFrame,
    intervals: list[dict],
    width: int,
    height: int,
    fps: float,
    max_step: float,
    strong_threshold: float,
) -> tuple[pd.DataFrame, int, int]:
    result = full_df.copy()
    result["SegmentId"] = -1
    result["ValidationStatus"] = ""
    result["RejectReason"] = ""
    result["MissingIntervalId"] = -1
    result["TileIndex"] = -1
    interval_map = {int(item["id"]): item for item in intervals}
    candidate_map = {
        int(frame): group.sort_values("Confidence", ascending=False).to_dict("records")
        for frame, group in candidates.groupby("Frame")
    } if not candidates.empty else {}

    max_step_px = scaled_max_step(max_step, width, height, fps)
    last_points: list[tuple[int, float, float]] = []
    segment_id = -1
    accepted_tile = 0
    rejected_tile = 0

    for idx, row in result.iterrows():
        frame = int(row["Frame"])
        if int(row["Visibility"]) == 1:
            if not last_points or frame != last_points[-1][0] + 1:
                segment_id += 1
            elif math.dist((last_points[-1][1], last_points[-1][2]), point(row)) > max_step_px * 2.0:
                segment_id += 1
            result.loc[idx, "SegmentId"] = segment_id
            result.loc[idx, "ValidationStatus"] = "full_frame_raw"
            last_points.append((frame, float(row["X"]), float(row["Y"])))
            last_points = last_points[-2:]
            continue

        chosen = None
        reject_reason = "missing"
        for cand in candidate_map.get(frame, []):
            interval_id = int(cand.get("MissingIntervalId", -1))
            ok, status, reason = temporal_accept(
                cand,
                last_points,
                interval_map.get(interval_id),
                max_step_px,
                strong_threshold,
            )
            if ok:
                chosen = (cand, status)
                break
            reject_reason = reason
            rejected_tile += 1

        if chosen is None:
            result.loc[idx, "Source"] = "missing"
            result.loc[idx, "RejectReason"] = reject_reason
            continue

        cand, status = chosen
        source = "adaptive_tile_raw" if float(cand["Confidence"]) >= strong_threshold else "adaptive_tile_weak_validated"
        if not last_points or frame != last_points[-1][0] + 1:
            segment_id += 1
        result.loc[idx, "Visibility"] = 1
        result.loc[idx, "X"] = round(float(cand["X"]))
        result.loc[idx, "Y"] = round(float(cand["Y"]))
        result.loc[idx, "Confidence"] = float(cand["Confidence"])
        result.loc[idx, "Source"] = source
        result.loc[idx, "SegmentId"] = segment_id
        result.loc[idx, "ValidationStatus"] = status
        result.loc[idx, "MissingIntervalId"] = int(cand["MissingIntervalId"])
        result.loc[idx, "TileIndex"] = int(cand["TileIndex"])
        last_points.append((frame, float(cand["X"]), float(cand["Y"])))
        last_points = last_points[-2:]
        accepted_tile += 1

    return result, accepted_tile, rejected_tile


def run_inpaint(model, seq_len, base_df: pd.DataFrame, width: int, height: int, batch_size: int, device):
    pred_dict = {
        "Frame": base_df["Frame"].astype(int).tolist(),
        "Visibility": base_df["Visibility"].astype(int).tolist(),
        "X": base_df["X"].astype(float).tolist(),
        "Y": base_df["Y"].astype(float).tolist(),
        "Img_scaler": (width / 512, height / 288),
        "Img_shape": (width, height),
    }
    pred_dict["Inpaint_Mask"] = generate_inpaint_mask(pred_dict, th_h=height * 0.05)
    dataset = Shuttlecock_Trajectory_Dataset(
        seq_len=seq_len,
        sliding_step=1,
        data_mode="coordinate",
        pred_dict=pred_dict,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    weight = get_ensemble_weight(seq_len, "weight")
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    coor_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
    sample_count = 0
    num_sample = len(dataset)
    outputs: dict[int, tuple[float, float, int]] = {}
    with torch.no_grad():
        for i, coor_pred, inpaint_mask in tqdm(loader, desc="InpaintNet"):
            coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
            coor_inpaint = model(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.0
            coor_buffer = torch.cat((coor_buffer, coor_inpaint), dim=0)
            b_size = int(i.shape[0])
            for b in range(b_size):
                if sample_count < buffer_size:
                    coor = coor_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    coor = (coor_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)
                frame = int(i[b][0][1])
                outputs[frame] = (float(coor[0] * width), float(coor[1] * height), 1)
                sample_count += 1
                if sample_count == num_sample:
                    coor_buffer = torch.cat((coor_buffer, torch.zeros((buffer_size, seq_len, 2))), dim=0)
                    for f in range(1, seq_len):
                        tail = coor_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        outputs[int(i[-1][f][1])] = (float(tail[0] * width), float(tail[1] * height), 1)
            coor_buffer = coor_buffer[-buffer_size:]
    return outputs


def apply_safe_inpaint(
    result: pd.DataFrame,
    inpaint_outputs: dict[int, tuple[float, float, int]],
    max_gap: int,
    max_deviation: float,
    max_step: float,
    width: int,
    height: int,
    fps: float,
) -> tuple[pd.DataFrame, int, int]:
    output = result.copy()
    frames = output["Frame"].astype(int).tolist()
    max_step_px = scaled_max_step(max_step, width, height, fps)
    filled = 0
    rejected_long = 0
    pos = 0
    while pos < len(frames):
        if int(output.iloc[pos]["Visibility"]) == 1:
            pos += 1
            continue
        start = pos
        while pos < len(frames) and int(output.iloc[pos]["Visibility"]) == 0:
            pos += 1
        end = pos - 1
        gap_len = end - start + 1
        if start == 0 or pos >= len(frames):
            continue
        if gap_len > max_gap:
            rejected_long += 1
            continue
        left = output.iloc[start - 1]
        right = output.iloc[pos]
        if int(left["Visibility"]) != 1 or int(right["Visibility"]) != 1:
            continue
        frame_span = int(right["Frame"]) - int(left["Frame"])
        if frame_span <= 0:
            continue
        left_point = point(left)
        right_point = point(right)
        accepted = []
        prev_point = left_point
        ok_gap = True
        for idx in range(start, end + 1):
            frame = frames[idx]
            pred = inpaint_outputs.get(frame)
            if pred is None:
                ok_gap = False
                break
            cand = (pred[0], pred[1])
            ratio = (frame - int(left["Frame"])) / frame_span
            expected = (
                left_point[0] + (right_point[0] - left_point[0]) * ratio,
                left_point[1] + (right_point[1] - left_point[1]) * ratio,
            )
            if math.dist(cand, expected) > max_deviation:
                ok_gap = False
                break
            if math.dist(prev_point, cand) > max_step_px:
                ok_gap = False
                break
            accepted.append((idx, cand))
            prev_point = cand
        if ok_gap and math.dist(prev_point, right_point) <= max_step_px:
            for idx, cand in accepted:
                output.loc[idx, "Visibility"] = 1
                output.loc[idx, "X"] = round(cand[0])
                output.loc[idx, "Y"] = round(cand[1])
                output.loc[idx, "Confidence"] = 0.0
                output.loc[idx, "Source"] = "inpaint_short_gap"
                output.loc[idx, "ValidationStatus"] = "inpaint_safe_short_gap"
                output.loc[idx, "SegmentId"] = int(left["SegmentId"])
                filled += 1
    return output, filled, rejected_long


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    profiler = TrackNetProfiler(args.profile)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(args.video_file).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with profiler.stage("video_metadata"):
        total_frames, fps, width, height = video_metadata(args.video_file, profiler)
    with profiler.stage("model_load"):
        tracknet, tracknet_seq_len, bg_mode = load_tracknet(args.tracknet_file, device, profiler)
    profiler.sample_memory("after_model_load")

    full_started = time.perf_counter()
    full_by_frame = {}
    segment_plan = segment_ranges(total_frames, args.segment_frames, args.segment_overlap)
    reader = VideoSegmentReader(args.video_file, profiler, "full_frame")
    try:
        segment_iterator = reader.segments(segment_plan)
        for segment_index, (start, end, segment) in enumerate(segment_iterator):
            segment_started = time.perf_counter()
            with profiler.stage("background_estimation"):
                segment_background = estimate_frame_background(segment, args.background_sample_frames, profiler) if bg_mode else None
            inference_before = profiler.timings.get("full_frame_inference", 0.0)
            detections = run_tracknet_heatmap(frames_bgr=segment, frame_ids=list(range(start, start + len(segment))), model=tracknet, seq_len=tracknet_seq_len, bg_mode=bg_mode, batch_size=args.batch_size, threshold=args.strong_threshold, device=device, progress_desc=f"Full-frame segment {segment_index + 1}/{len(segment_plan)}", background_sample_frames=args.background_sample_frames, profiler=profiler, profile_prefix="full_frame", background_rgb=segment_background)
            merge_started = time.perf_counter()
            for detection in detections:
                previous = full_by_frame.get(detection.frame)
                if previous is None or detection.confidence > previous.confidence:
                    full_by_frame[detection.frame] = detection
            merge_sec = time.perf_counter() - merge_started
            profiler.add_time("overlap_merge", merge_sec)
            profiler.add_count("full_frame_processed_frames", len(segment))
            profiler.segments.append({"segment_index": segment_index, "start_frame": start, "end_frame": end - 1, "frame_count": len(segment), "overlap_frame_count": 0 if segment_index == 0 else min(args.segment_overlap, len(segment)), "inference_sec": profiler.timings.get("full_frame_inference", 0.0) - inference_before, "merge_sec": merge_sec, "elapsed_sec": time.perf_counter() - segment_started, "detected_frame_count": sum(d.visibility for d in detections)})
            profiler.sample_memory(f"segment_{segment_index}_end")
    finally:
        reader.close()
    full_dets = [full_by_frame.get(frame, HeatmapDetection(frame, 0, 0.0, 0.0, 0.0)) for frame in range(total_frames)]
    elapsed_full = time.perf_counter() - full_started
    full_df = pd.DataFrame(
        {
            "Frame": [det.frame for det in full_dets],
            "Visibility": [det.visibility for det in full_dets],
            "X": [round(det.x) for det in full_dets],
            "Y": [round(det.y) for det in full_dets],
            "Confidence": [det.confidence for det in full_dets],
            "Source": ["full_frame_raw" if det.visibility else "missing" for det in full_dets],
        }
    )
    full_csv = save_dir / f"{video_stem}_full_frame_raw.csv"
    with profiler.stage("csv_write"):
        full_df.to_csv(full_csv, index=False)

    with profiler.stage("missing_interval_detection"):
        intervals = find_missing_intervals(full_df)
    intervals_json = save_dir / f"{video_stem}_missing_intervals.json"
    intervals_json.write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    ranges = merge_supplement_ranges(intervals, total_frames, args.context_frames, args.tiling_mode)
    tiles = generate_adaptive_tiles(width, height, args.max_tiles)
    tile_started = time.perf_counter()
    candidate_rows: list[dict] = []
    tile_inference_frame_count = 0

    if args.tiling_mode != "off":
        tile_reader = VideoRangeReader(args.video_file, profiler)
        for supplement_range in ranges:
            range_started = time.perf_counter()
            range_tile_count = 0
            range_frames = list(range(supplement_range["start"], supplement_range["end"] + 1))
            active_tiles: dict[int, Tile] = {}
            for interval_id in supplement_range["interval_ids"]:
                for tile in choose_tiles_for_interval(
                    tiles,
                    intervals[interval_id],
                    width,
                    height,
                    args.corridor_margin,
                ):
                    active_tiles[tile.index] = tile
            with profiler.stage("tile_video_decode"):
                range_buffer = tile_reader.read(range_frames[0], range_frames[-1] + 1)
            with profiler.stage("background_estimation"):
                range_background = estimate_frame_background(range_buffer, args.background_sample_frames, profiler) if bg_mode else None
            for tile in active_tiles.values():
                range_tile_count += 1
                with profiler.stage("tile_crop_generation"):
                    tile_frames = [frame[tile.y1:tile.y2, tile.x1:tile.x2] for frame in range_buffer]
                profiler.add_count("tile_crop_count", len(tile_frames))
                detections = run_tracknet_heatmap(
                    frames_bgr=tile_frames,
                    frame_ids=range_frames,
                    model=tracknet,
                    seq_len=tracknet_seq_len,
                    bg_mode=bg_mode,
                    batch_size=args.batch_size,
                    threshold=args.candidate_threshold,
                    device=device,
                    background_sample_frames=args.background_sample_frames,
                    offset=(tile.x1, tile.y1),
                    progress_desc=f"Tile {tile.index}",
                    profiler=profiler,
                    profile_prefix="tile",
                    background_rgb=None if range_background is None else range_background[tile.y1:tile.y2, tile.x1:tile.x2],
                )
                tile_inference_frame_count += len(tile_frames)
                del tile_frames
                for det in detections:
                    if det.confidence < args.candidate_threshold:
                        continue
                    interval_id = nearest_interval_id(det.frame, intervals, supplement_range["interval_ids"])
                    candidate_rows.append(
                        {
                            "Frame": det.frame,
                            "X": round(det.x),
                            "Y": round(det.y),
                            "Confidence": det.confidence,
                            "Source": "adaptive_tile_candidate",
                            "TileIndex": tile.index,
                            "TileX1": tile.x1,
                            "TileY1": tile.y1,
                            "TileX2": tile.x2,
                            "TileY2": tile.y2,
                            "MissingIntervalId": interval_id,
                            "CandidateStatus": "strong" if det.confidence >= args.strong_threshold else "weak",
                            "RejectReason": "",
                        }
                    )
            del range_buffer
            profiler.tile_intervals.append({"start_frame": supplement_range["start"], "end_frame": supplement_range["end"], "interval_ids": supplement_range["interval_ids"], "tile_count": range_tile_count, "processed_frames": len(range_frames) * range_tile_count, "elapsed_sec": time.perf_counter() - range_started})
        tile_reader.close()

    elapsed_tile = time.perf_counter() - tile_started
    profiler.add_count("missing_interval_count", len(intervals))
    profiler.add_count("missing_interval_total_frames", sum(x["length"] for x in intervals))
    profiler.add_count("tile_recovery_interval_count", len(ranges))
    profiler.add_count("tile_recovery_processed_frames", tile_inference_frame_count)
    profiler.sample_memory("after_tile_recovery")
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = dedupe_candidates(candidates)
    else:
        candidates = pd.DataFrame(
            columns=[
                "Frame", "X", "Y", "Confidence", "Source", "TileIndex", "TileX1", "TileY1",
                "TileX2", "TileY2", "MissingIntervalId", "CandidateStatus", "RejectReason",
                "merged_candidate_count", "merged_tile_indices",
            ]
        )
    candidates_csv = save_dir / f"{video_stem}_adaptive_candidates.csv"
    with profiler.stage("csv_write"):
        candidates.to_csv(candidates_csv, index=False)

    with profiler.stage("tile_result_merge"):
        result, accepted_tile, rejected_tile = fuse_results(full_df, candidates, intervals, width, height, fps, args.max_step, args.strong_threshold)

    inpaint_count = 0
    rejected_long_inpaint = 0
    inpaint_started = time.perf_counter()
    if args.inpaintnet_file:
        with profiler.stage("inpaint_model_load"):
            inpaintnet, inpaint_seq_len = load_inpaintnet(args.inpaintnet_file, device, profiler)
        with profiler.stage("short_gap_inpaint", synchronize_cuda=True):
            inpaint_outputs = run_inpaint(inpaintnet, inpaint_seq_len, result, width, height, args.batch_size, device)
        result, inpaint_count, rejected_long_inpaint = apply_safe_inpaint(
            result,
            inpaint_outputs,
            args.max_inpaint_gap,
            args.max_deviation,
            args.max_step,
            width,
            height,
            fps,
        )
    elapsed_inpaint = time.perf_counter() - inpaint_started

    final_columns = [
        "Frame", "Visibility", "X", "Y", "Confidence", "Source", "SegmentId",
        "ValidationStatus", "RejectReason", "MissingIntervalId", "TileIndex",
    ]
    result = result[final_columns]
    result["Source"] = result["Source"].where(result["Visibility"].astype(int) == 1, "missing")
    safe_csv = save_dir / f"{video_stem}_ball_adaptive_safe.csv"
    with profiler.stage("csv_write"):
        result.to_csv(safe_csv, index=False)

    debug = {
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processing_mode": "chunked",
        "segment_frames": args.segment_frames,
        "segment_overlap": args.segment_overlap,
        "segment_count": len(segment_plan),
        "background_sample_frames": min(args.background_sample_frames, args.segment_frames),
        "aspect_ratio": width / height,
        "full_frame_visible_count": int((full_df["Visibility"].astype(int) == 1).sum()),
        "full_frame_visibility_ratio": float((full_df["Visibility"].astype(int) == 1).mean()),
        "missing_interval_count": len(intervals),
        "missing_intervals": intervals,
        "adaptive_tiling_triggered": bool(ranges),
        "tile_layouts": [asdict(tile) for tile in tiles],
        "tile_inference_ranges": ranges,
        "tile_inference_frame_count": tile_inference_frame_count,
        "tile_candidate_count": int(len(candidates)),
        "tile_strong_candidate_count": int((candidates["Confidence"].astype(float) >= args.strong_threshold).sum()) if not candidates.empty else 0,
        "tile_weak_candidate_count": int(((candidates["Confidence"].astype(float) >= args.candidate_threshold) & (candidates["Confidence"].astype(float) < args.strong_threshold)).sum()) if not candidates.empty else 0,
        "tile_accepted_count": accepted_tile,
        "tile_rejected_count": rejected_tile,
        "merged_candidate_count": int(candidates["merged_candidate_count"].sum()) if "merged_candidate_count" in candidates else 0,
        "inpaint_short_gap_count": inpaint_count,
        "rejected_long_inpaint_gap_count": rejected_long_inpaint,
        "final_visible_count": int((result["Visibility"].astype(int) == 1).sum()),
        "final_visibility_ratio": float((result["Visibility"].astype(int) == 1).mean()),
        "trajectory_segment_count": int(result.loc[result["SegmentId"].astype(int) >= 0, "SegmentId"].nunique()),
        "elapsed_full_frame_sec": elapsed_full,
        "elapsed_tile_sec": elapsed_tile,
        "full_frame_elapsed_sec": elapsed_full,
        "tile_recovery_elapsed_sec": elapsed_tile,
        "inpaint_elapsed_sec": elapsed_inpaint,
        "render_elapsed_sec": 0.0,
        "elapsed_total_sec": time.perf_counter() - started,
        "outputs": {
            "full_frame_raw_csv": str(full_csv),
            "missing_intervals_json": str(intervals_json),
            "adaptive_candidates_csv": str(candidates_csv),
            "ball_adaptive_safe_csv": str(safe_csv),
        },
    }
    debug_json = save_dir / f"{video_stem}_adaptive_debug.json"
    debug_json.write_text(json.dumps(debug, indent=2), encoding="utf-8")

    if args.profile:
        profiler.add_count("unique_source_frames", total_frames)
        profiler.add_count("overlap_reprocessed_frames", sum(max(0, item[1] - item[0]) for item in segment_plan) - total_frames)
        profiler.add_count("detected_frame_count", debug["final_visible_count"])
        profiler.add_count("tile_inference_frame_times", tile_inference_frame_count)
        profiler.counts.setdefault("background_source_decode_count", 0)
        profiler.counts["total_source_decode_count"] = (
            profiler.counts.get("background_source_decode_count", 0)
            + profiler.counts.get("full_frame_source_decode_count", 0)
            + profiler.counts.get("tile_source_decode_count", 0)
        )
        profiler.finalize(save_dir / f"{video_stem}_performance_profile.json", device=device, video={"path": args.video_file, "width": width, "height": height, "fps": fps, "frame_count": total_frames, "duration_sec": total_frames / fps if fps else 0}, configuration={"segment_frames": args.segment_frames, "segment_overlap": args.segment_overlap, "segment_count": len(segment_plan), "batch_size": args.batch_size, "background_sample_frames": args.background_sample_frames, "tiling_mode": args.tiling_mode, "max_tiles": args.max_tiles})

    print("device =", device)
    print("full_frame_visible_count =", debug["full_frame_visible_count"])
    print("missing_interval_count =", debug["missing_interval_count"])
    print("tile_layouts =", debug["tile_layouts"])
    print("tile_inference_frame_count =", tile_inference_frame_count)
    print("tile_accepted_count =", accepted_tile)
    print("inpaint_short_gap_count =", inpaint_count)
    print("final_visible_count =", debug["final_visible_count"])
    print("trajectory_segment_count =", debug["trajectory_segment_count"])
    print("elapsed_total_sec =", debug["elapsed_total_sec"])
    print("safe_csv =", safe_csv)


if __name__ == "__main__":
    main()
