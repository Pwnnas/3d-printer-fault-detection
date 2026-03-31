"""
Timelapse Frame Extractor — for YOLOv8n dataset preparation
Extracts frames from 3D printer timelapse videos:
  • A few frames from the START  (clean print / empty bed reference)
  • A few frames from the MIDDLE (print progress context)
  • Many frames from the END     (where failures happen)

Usage:
  python extract_frames.py --input ./timelapses --output ./frames
  python extract_frames.py --input ./timelapses --output ./frames --start 3 --middle 3 --end 20
"""

import argparse
import os
import cv2
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def extract_frames(
    video_path: Path,
    out_dir: Path,
    n_start: int,
    n_middle: int,
    n_end: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path.name}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    secs  = total / fps

    print(f"  {video_path.name} — {total} frames, {secs:.0f}s")

    if total < (n_start + n_middle + n_end):
        print(f"  [WARN] Video too short, extracting every frame instead.")
        indices = list(range(total))
    else:
        # START: spread across first 8% of video
        start_region = max(1, int(total * 0.08))
        start_step   = max(1, start_region // n_start)
        start_frames = [i * start_step for i in range(n_start)]

        # MIDDLE: spread across 40–60% of video (shows print progress)
        mid_center  = total // 2
        mid_spread  = int(total * 0.10)
        mid_step    = max(1, (mid_spread * 2) // n_middle)
        mid_frames  = [mid_center - mid_spread + i * mid_step for i in range(n_middle)]

        # END: spread across last 15% of video (failures happen here)
        end_region  = max(1, int(total * 0.15))
        end_start   = total - end_region
        end_step    = max(1, end_region // n_end)
        end_frames  = [end_start + i * end_step for i in range(n_end)]
        # Always include the very last frame
        if (total - 1) not in end_frames:
            end_frames.append(total - 1)

        indices = sorted(set(start_frames + mid_frames + end_frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Label which section this frame came from
        pct = idx / max(total - 1, 1)
        if pct < 0.15:
            section = "start"
        elif pct < 0.65:
            section = "mid"
        else:
            section = "end"

        filename = f"{video_path.stem}__{section}_f{idx:05d}.jpg"
        cv2.imwrite(str(out_dir / filename), frame)
        saved += 1

    cap.release()
    print(f"  → Saved {saved} frames to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Extract start/middle/end frames from timelapse videos.")
    parser.add_argument("--input",  "-i", required=True,  help="Folder containing timelapse videos")
    parser.add_argument("--output", "-o", default="frames", help="Output folder (default: ./frames)")
    parser.add_argument("--start",  type=int, default=3,  help="Frames from start  (default: 3)")
    parser.add_argument("--middle", type=int, default=3,  help="Frames from middle (default: 3)")
    parser.add_argument("--end",    type=int, default=15, help="Frames from end    (default: 15)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    videos = [f for f in input_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(videos)} video(s). Extracting frames...\n")

    for video in sorted(videos):
        # Each video gets its own subfolder so you can browse by timelapse
        video_out = output_dir / video.stem
        extract_frames(video, video_out, args.start, args.middle, args.end)

    print(f"\nDone. Open {output_dir}/ and sort frames into class folders for Roboflow.")
    print("Suggested folders: spaghetti/ layer_shift/ warping/ blob/ empty/")


if __name__ == "__main__":
    main()
