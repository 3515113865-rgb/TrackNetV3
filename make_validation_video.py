from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TrackNet validation video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trail-length", type=int, default=6)
    parser.add_argument("--max-step", type=float, default=120.0)
    return parser.parse_args()


def source_color(source: str) -> tuple[int, int, int]:
    colors = {
        "full_frame_raw": (0, 0, 255),
        "adaptive_tile_raw": (255, 0, 0),
        "adaptive_tile_weak_validated": (255, 255, 0),
        "inpaint_short_gap": (0, 255, 0),
        "raw_tracknet": (0, 0, 255),
    }
    return colors.get(str(source), (180, 180, 180))


def numeric(row, name: str, default: float = 0.0) -> float:
    if row is None or name not in row:
        return default
    try:
        return float(row[name])
    except Exception:
        return default


def text(row, name: str, default: str = "") -> str:
    if row is None or name not in row:
        return default
    value = row[name]
    if pd.isna(value):
        return default
    return str(value)


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not video_path.exists():
        raise SystemExit(f"Video does not exist: {video_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV does not exist: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    required_columns = {"Frame", "Visibility", "X", "Y"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"CSV missing columns: {sorted(missing_columns)}")

    frame_map = {int(row["Frame"]): row for _, row in df.iterrows()}
    csv_min_frame = int(df["Frame"].min())
    frame_number_offset = 0 if csv_min_frame == 0 else 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Cannot create output video: {output_path}")

    trail: deque[tuple[int, tuple[int, int], int]] = deque(
        maxlen=max(2, args.trail_length)
    )

    frame_index = 0
    visible_frames = 0
    rejected_jump_count = 0
    broken_by_missing_count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        csv_frame_index = frame_index + frame_number_offset
        row = frame_map.get(csv_frame_index)

        visibility = int(numeric(row, "Visibility", 0.0))
        raw_x = int(round(numeric(row, "X", 0.0)))
        raw_y = int(round(numeric(row, "Y", 0.0)))
        source = text(row, "Source", "missing")
        confidence = numeric(row, "Confidence", 0.0)
        segment_id = int(numeric(row, "SegmentId", -1.0))
        missing_interval_id = int(numeric(row, "MissingIntervalId", -1.0))

        point_is_valid = (
            visibility == 1
            and 0 <= raw_x < width
            and 0 <= raw_y < height
            and source != "missing"
        )

        if point_is_valid:
            visible_frames += 1
            raw_point = (raw_x, raw_y)

            if trail:
                previous_frame, previous_point, previous_segment = trail[-1]
                if (
                    frame_index - previous_frame != 1
                    or previous_segment != segment_id
                    or segment_id < 0
                ):
                    trail.clear()
                elif math.dist(previous_point, raw_point) > args.max_step:
                    trail.clear()
                    rejected_jump_count += 1

            trail.append((frame_index, raw_point, segment_id))
            color = source_color(source)

            cv2.circle(frame, raw_point, 16, color, 2)
            cv2.line(frame, (raw_x - 22, raw_y), (raw_x - 8, raw_y), color, 2)
            cv2.line(frame, (raw_x + 8, raw_y), (raw_x + 22, raw_y), color, 2)
            cv2.line(frame, (raw_x, raw_y - 22), (raw_x, raw_y - 8), color, 2)
            cv2.line(frame, (raw_x, raw_y + 8), (raw_x, raw_y + 22), color, 2)

            trail_points = list(trail)
            for index in range(1, len(trail_points)):
                previous_frame, previous_point, previous_segment = trail_points[index - 1]
                current_frame, current_point, current_segment = trail_points[index]
                if (
                    current_frame - previous_frame == 1
                    and previous_segment == current_segment
                    and current_segment >= 0
                    and math.dist(previous_point, current_point) <= args.max_step
                ):
                    cv2.line(
                        frame,
                        previous_point,
                        current_point,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        else:
            if trail:
                broken_by_missing_count += 1
            trail.clear()

        status_text = (
            f"Frame {frame_index}/{total_frames} | V={visibility} | "
            f"Source={source} | Conf={confidence:.3f} | "
            f"Seg={segment_id} | Gap={missing_interval_id}"
        )
        cv2.rectangle(frame, (10, 10), (920, 58), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status_text,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()

    print("video =", video_path)
    print("csv =", csv_path)
    print("output =", output_path.resolve())
    print("written_frames =", frame_index)
    print("visible_frames =", visible_frames)
    print("rejected_jump_count =", rejected_jump_count)
    print("broken_by_missing_count =", broken_by_missing_count)
    print("trail_length =", args.trail_length)
    print("max_step =", args.max_step)
    print("done")


if __name__ == "__main__":
    main()
