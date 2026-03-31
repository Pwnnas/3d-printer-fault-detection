"""
Auto-Labeler — Vision model pre-sorts extracted frames into class folders.
Uses Ollama (local) to classify each frame, then copies it into the matching
subfolder. Uncertain or ambiguous frames go to uncertain/ for manual review.

Usage:
  python autolabel_frames.py --input ./frames --output ./labeled
  python autolabel_frames.py --input ./frames --output ./labeled --model llama3.2-vision:11b
  python autolabel_frames.py --input ./frames --output ./labeled --model llama3.2-vision

Workflow after this script:
  1. Review uncertain/ and move to correct class folder
  2. Spot-check the class folders using review.csv
  3. Upload labeled/ to Roboflow and draw bounding boxes
"""

import argparse
import base64
import csv
import os
import shutil
import time
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

VALID_CLASSES = {"spaghetti", "layer_shift", "warping", "blob", "empty", "ok", "uncertain"}

SYSTEM_PROMPT = """\
***IF NOT 100% SURE THEN PICK UNCERTAIN***

You are a 3D printing fault detection classifier. Your ONLY job is to look at a photo from a 3D printer camera and respond with exactly ONE word from this list:

  spaghetti  - tangled filament mess, failed print, stringing disaster
  layer_shift - layers are offset/misaligned, or clear gaps from under-extrusion
  warping    - edges of the print are lifting off the bed
  blob       - obvious lump, zit, or over-extrusion blob on the print surface
  empty      - the print bed is empty, no print present
  ok         - print looks normal and healthy
  uncertain  - anything unclear, ambiguous, or that does not fit the above

***IF NOT 100% SURE THEN PICK UNCERTAIN***

Rules:
- Respond with ONLY the single word. No punctuation, no explanation, no extra text.
- If you see multiple issues, pick the most severe one (spaghetti > layer_shift > warping > blob).
- When in doubt: uncertain.
"""


def classify_image(image_path: Path, model: str, server_url: str) -> tuple[str, str]:
    """
    Send image to Ollama vision model and return (class, raw_response).
    Falls back to 'uncertain' on any error.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        r = requests.post(server_url, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip().lower()
        # Extract just the first word in case the model still adds extra text
        first_word = raw.split()[0].rstrip(".,!?:") if raw else "uncertain"
        label = first_word if first_word in VALID_CLASSES else "uncertain"
        return label, raw
    except Exception as e:
        return "uncertain", f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Auto-label frames using a local Ollama vision model.")
    parser.add_argument("--input",  "-i", required=True,        help="Folder of extracted frames (can have subfolders)")
    parser.add_argument("--output", "-o", default="labeled",    help="Output folder (default: ./labeled)")
    parser.add_argument("--model",  "-m", default="llama3.2-vision:11b", help="Ollama model (default: llama3.2-vision:11b)")
    parser.add_argument("--server", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--delay",  type=float, default=0.2,    help="Seconds between requests (default: 0.2)")
    args = parser.parse_args()

    server_url = f"{args.server.rstrip('/')}/v1/chat/completions"
    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    # Collect all images recursively
    images = sorted([
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        print(f"No images found in {input_dir}")
        return

    # Create output class folders
    for cls in VALID_CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    print(f"Model  : {args.model}")
    print(f"Server : {server_url}")
    print(f"Images : {len(images)}")
    print(f"Output : {output_dir}/\n")

    csv_path = output_dir / "review.csv"
    counts   = {cls: 0 for cls in VALID_CLASSES}

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label", "raw_response"])

        for i, img_path in enumerate(images, 1):
            print(f"[{i:>4}/{len(images)}] {img_path.name}", end=" → ", flush=True)

            label, raw = classify_image(img_path, args.model, server_url)
            print(label)

            dest = output_dir / label / img_path.name
            # Avoid overwriting if two videos have a frame with the same name
            if dest.exists():
                dest = output_dir / label / f"{img_path.stem}_{i}{img_path.suffix}"
            shutil.copy2(img_path, dest)

            counts[label] += 1
            writer.writerow([img_path.name, label, raw])

            if args.delay:
                time.sleep(args.delay)

    print("\n── Summary ──────────────────────────────")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count:
            bar = "█" * min(count, 40)
            print(f"  {cls:<12} {count:>4}  {bar}")
    print(f"\nReview log: {csv_path}")
    print(f"Check uncertain/ first, then spot-check the class folders.")


if __name__ == "__main__":
    main()
