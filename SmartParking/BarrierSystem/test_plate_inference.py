"""
Test inference: plate detection + OCR pipeline on user-supplied images.

Runs the same PlateRecognizer used by the barrier controller (YOLO plate
detector + ParseQ OCR) on arbitrary images so you can eyeball the output
before wiring up cameras.

Usage:
    python test_plate_inference.py <image_or_folder>
    python test_plate_inference.py ./samples/
    python test_plate_inference.py ./samples/car1.png

Keys (inside the OpenCV window):
    N / -> / Space   next image
    P / <-           previous image
    R                re-run on current image (re-runs OCR)
    E                toggle brightness/saturation enhancement
    S                save annotated image to ./test_results/
    Q / Esc          quit
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# --- Make the web_app PlateRecognizer importable ---
WEB_APP_DIR = Path(__file__).parent.parent / "web_app"
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

from plate_scanner import PlateRecognizer  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW_NAME = "Plate Inference Test"
RESULTS_DIR = Path(__file__).parent / "test_results"


def collect_images(target: Path) -> list[Path]:
    """Return a sorted list of images from a single file or directory."""
    if target.is_file():
        return [target] if target.suffix.lower() in IMAGE_EXTS else []
    if target.is_dir():
        imgs = [p for p in sorted(target.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return imgs
    return []


def run_inference(recognizer: PlateRecognizer, img_bgr: np.ndarray) -> dict:
    """Run detection + OCR on the full image (no car bbox hint).

    Returns a dict with timing and the recognizer result (or None)."""
    t0 = time.time()
    result = recognizer.detect_plate(img_bgr, car_bbox=None)
    elapsed = time.time() - t0
    return {
        "result": result,
        "elapsed_ms": elapsed * 1000.0,
    }


def annotate(img_bgr: np.ndarray, info: dict, filename: str,
             index: int, total: int) -> np.ndarray:
    """Draw bbox + OCR text + status bar + cropped-plate preview on a copy."""
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    result = info["result"]

    # Top status bar
    bar_h = 40
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), -1)
    status = f"[{index + 1}/{total}]  {filename}   |  OCR: {info['elapsed_ms']:.0f} ms"
    cv2.putText(canvas, status, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Bottom banner with OCR result
    banner_h = 60
    py = h - banner_h
    cv2.rectangle(canvas, (0, py), (w, h), (20, 20, 20), -1)

    if result is None:
        cv2.putText(canvas, "NO PLATE DETECTED", (10, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas

    # Draw plate bbox
    x1, y1, x2, y2 = result["bbox"]
    det_conf = result["confidence"]
    plate_text = result["plate_text"] or "<empty>"

    color = (0, 255, 0) if result["plate_text"] else (0, 165, 255)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    tag = f"{plate_text}  ({det_conf * 100:.0f}%)"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(th + 6, y1 - 6)
    cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), (0, 0, 0), -1)
    cv2.putText(canvas, tag, (x1 + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Banner text
    banner_text = f"PLATE: {plate_text}   |   det conf {det_conf * 100:.1f}%"
    cv2.putText(canvas, banner_text, (10, py + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Cropped plate preview in top-right corner
    plate_img = result.get("plate_image")
    if plate_img is not None and plate_img.size > 0:
        target_w = 260
        ph, pw = plate_img.shape[:2]
        if pw > 0 and ph > 0:
            scale = target_w / pw
            preview = cv2.resize(plate_img, (target_w, max(int(ph * scale), 20)),
                                 interpolation=cv2.INTER_CUBIC)
            ph2, pw2 = preview.shape[:2]
            ox = w - pw2 - 10
            oy = bar_h + 10
            if oy + ph2 < py - 10:
                cv2.rectangle(canvas, (ox - 3, oy - 3),
                              (ox + pw2 + 3, oy + ph2 + 3), color, 2)
                canvas[oy:oy + ph2, ox:ox + pw2] = preview

    return canvas


def save_annotated(canvas: np.ndarray, src_name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = Path(src_name).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{stem}_{ts}.png"
    cv2.imwrite(str(out), canvas)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plate detection + OCR tester")
    parser.add_argument("path", help="Image file or directory of images")
    parser.add_argument("--model", default=None,
                        help="Optional path to a YOLO plate-detector .pt")
    parser.add_argument("--conf", type=float, default=0.06,
                        help="YOLO confidence threshold (default 0.06 — PlateScanner's own default)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable test-time augmentation (default: on, matches upstream)")
    parser.add_argument("--no-match-train-aug", action="store_true",
                        help="Disable ToGray+CLAHE preprocessing (default: on, matches training)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable brightness/saturation boost on input image")
    parser.add_argument("--brightness", type=float, default=1.35,
                        help="Brightness multiplier for enhancement (default 1.35)")
    parser.add_argument("--saturation", type=float, default=1.4,
                        help="Saturation multiplier for enhancement (default 1.4)")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    images = collect_images(target)
    if not images:
        print(f"No images found at: {target}")
        print(f"Supported extensions: {sorted(IMAGE_EXTS)}")
        sys.exit(1)

    print(f"Found {len(images)} image(s).")
    print(f"Config: conf={args.conf}  augment={not args.no_augment}  "
          f"match_train_aug={not args.no_match_train_aug}  "
          f"enhance={not args.no_enhance}  "
          f"brightness={args.brightness}  saturation={args.saturation}")
    print("Loading PlateRecognizer (YOLO + ParseQ, eager warmup)...")
    recognizer = PlateRecognizer(
        plate_model_path=args.model,
        confidence=args.conf,
        augment=not args.no_augment,
        match_training_aug=not args.no_match_train_aug,
        enhance=not args.no_enhance,
        brightness=args.brightness,
        saturation=args.saturation,
        eager_load=True,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 800)

    idx = 0
    cache: dict[int, dict] = {}

    def ensure(i: int, img: np.ndarray) -> dict:
        if i not in cache:
            info = run_inference(recognizer, img)
            r = info["result"]
            plate = (r["plate_text"] if r else None) or "<none>"
            conf = (r["confidence"] if r else 0.0)
            print(f"[{i + 1}/{len(images)}] {images[i].name}  "
                  f"enhance={recognizer.enhance}  plate={plate!r}  "
                  f"conf={conf:.2f}  time={info['elapsed_ms']:.0f} ms")
            cache[i] = info
        return cache[i]

    while True:
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path} — skipping")
            images.pop(idx)
            cache.pop(idx, None)
            if not images:
                print("No readable images left.")
                break
            idx = idx % len(images)
            continue

        info = ensure(idx, img)
        # PlateRecognizer applies enhance internally — show what it sees by
        # mirroring the same boost on our display copy.
        display_img = recognizer._enhance_frame(img) if recognizer.enhance else img
        annotated = annotate(display_img, info, img_path.name, idx, len(images))
        if recognizer.enhance:
            cv2.putText(annotated,
                        f"ENHANCED b={recognizer.brightness:.2f} s={recognizer.saturation:.2f}",
                        (10, annotated.shape[0] - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(0) & 0xFF

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key in (ord("q"), 27):          # Q / Esc
            break
        elif key in (ord("n"), ord(" "), 83):   # N / Space / Right arrow
            idx = (idx + 1) % len(images)
        elif key in (ord("p"), 81):        # P / Left arrow
            idx = (idx - 1) % len(images)
        elif key == ord("r"):              # Re-run on current image
            cache.pop(idx, None)
        elif key == ord("e"):              # Toggle enhancement
            recognizer.enhance = not recognizer.enhance
            cache.clear()
            print(f"[enhance] {'ON' if recognizer.enhance else 'OFF'}")
        elif key == ord("s"):              # Save annotated image
            out = save_annotated(annotated, img_path.name)
            print(f"Saved: {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
