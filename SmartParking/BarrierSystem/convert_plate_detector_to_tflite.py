"""
Convert the YOLO11x license-plate detector from PyTorch (.pt) to TensorFlow Lite
(.tflite) with FP16 quantization.

Why FP16 TFLite:
  - File size drops by ~2x vs FP32 (and by ~4x vs the original PyTorch .pt because
    the .pt also stores optimizer state / metadata not needed for inference).
  - FP16 inference is faster on most modern CPUs and dramatically faster on
    accelerators that support it (Coral Edge TPU, ARM NEON, mobile NPUs).
  - Demonstrates that the system is deployable in a real-time / edge setting,
    not just on a desktop GPU.

Usage:
    python convert_plate_detector_to_tflite.py

Optional flags:
    --imgsz 640       Inference image size (must match training; YOLO11 default 640).
    --no-benchmark    Skip the speed comparison at the end.
    --runs 30         Number of timed inference iterations per model.

Extra dependencies (Ultralytics exporter pulls in TF/ONNX toolchain):
    pip install tensorflow onnx onnx2tf onnxslim sng4onnx tflite_support
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PT_MODEL = PROJECT_ROOT / "Models" / "plate_scanner" / "yolo11x.pt"
EXPORT_DIR = PROJECT_ROOT / "Models" / "plate_scanner"


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def export_pt_to_onnx(pt_path: Path, imgsz: int) -> Path:
    """First leg of the chain: PT → ONNX via Ultralytics.

    We do NOT reuse an existing .onnx unless we can confirm it was built at the
    requested imgsz — a 320-shape ONNX silently passing for a 640 export is the
    kind of bug that's only caught when inference results look wrong. Cheap to
    re-export, expensive to debug.
    """
    import onnx as _onnx
    from ultralytics import YOLO

    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists() and onnx_path.stat().st_mtime > pt_path.stat().st_mtime:
        try:
            existing = _onnx.load(str(onnx_path))
            in_shape = [d.dim_value for d in existing.graph.input[0].type.tensor_type.shape.dim]
            if len(in_shape) >= 4 and in_shape[2] == imgsz and in_shape[3] == imgsz:
                print(f"[convert] Reusing existing ONNX (imgsz={imgsz}): {onnx_path}")
                return onnx_path
            print(f"[convert] Existing ONNX has shape {in_shape}, requested imgsz={imgsz} — re-exporting")
        except Exception as e:
            print(f"[convert] Couldn't validate existing ONNX ({e}) — re-exporting")

    print(f"[convert] Loading model: {pt_path}")
    model = YOLO(str(pt_path))
    print(f"[convert] Exporting PT → ONNX (imgsz={imgsz})...")
    onnx_str = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=19)
    return Path(onnx_str)


def _run_in_subprocess(stage_name: str, code: str) -> None:
    """Run a Python snippet in a fresh subprocess so its peak memory doesn't
    stack on top of the parent process. Critical for the YOLO11x conversion:
    PT→ONNX (torch+ultralytics) and ONNX→TF (tensorflow+onnx2tf) each peak
    around 4–6 GB, and 16 GB systems can't hold both at once."""
    env = os.environ.copy()
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print(f"[convert] >>> {stage_name} (subprocess)")
    proc = subprocess.run([sys.executable, "-c", code], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{stage_name} subprocess failed (exit {proc.returncode})")
    print(f"[convert] <<< {stage_name} done")


def export_onnx_to_tflite(onnx_path: Path, imgsz: int) -> Path:
    """Second leg: ONNX → TFLite (FP16) in two subprocess steps.

    Step A: onnx2tf produces a TF SavedModel from the ONNX. We tolerate a
    non-zero exit if the SavedModel still lands on disk — onnx2tf occasionally
    crashes during its own FP16 pass even after the SavedModel is written.

    Step B: TF's `TFLiteConverter.from_saved_model` builds the FP16 .tflite
    from the SavedModel. More reliable than onnx2tf's own FP16 path.

    Both steps run as fresh subprocesses to keep peak RAM low — the YOLO11x
    model is too large for a single Python process to hold all of torch +
    ultralytics + tensorflow + onnx2tf simultaneously on a 16 GB system.
    """
    saved_model_dir = EXPORT_DIR / "yolo11x_saved_model"
    saved_model_pb = saved_model_dir / "saved_model.pb"

    if not saved_model_pb.exists():
        # Build the TF SavedModel via onnx2tf in its own process.
        # (We can't `try`/`except` a SIGSEGV from inside Python — must isolate.)
        snippet = textwrap.dedent(f"""
            import onnx2tf
            try:
                onnx2tf.convert(
                    input_onnx_file_path={str(onnx_path)!r},
                    output_folder_path={str(saved_model_dir)!r},
                    copy_onnx_input_output_names_to_tflite=True,
                    non_verbose=True,
                    output_signaturedefs=True,
                )
            except SystemExit:
                pass
        """)
        try:
            _run_in_subprocess("onnx2tf SavedModel build", snippet)
        except RuntimeError as e:
            # onnx2tf may exit non-zero even after producing the SavedModel.
            print(f"[convert] {e} — checking SavedModel anyway")

    if not saved_model_pb.exists():
        raise RuntimeError(
            f"onnx2tf failed to produce {saved_model_pb}. "
            f"Likely OOM — close other apps and retry."
        )
    print(f"[convert] SavedModel ready: {saved_model_dir}")

    # FP16 quantization via TF in a fresh subprocess. Writes the .tflite to disk
    # and exits, so peak RAM falls back to ~zero before we proceed.
    target = EXPORT_DIR / f"{onnx_path.stem}_float16.tflite"
    snippet = textwrap.dedent(f"""
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model({str(saved_model_dir)!r})
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()
        with open({str(target)!r}, "wb") as f:
            f.write(tflite_bytes)
        print(f"[tf-quant] Wrote {{len(tflite_bytes)/1024/1024:.1f}} MB to {{{str(target)!r}}}")
    """)
    _run_in_subprocess("TF FP16 quantization", snippet)

    if not target.exists():
        raise RuntimeError(f"TF FP16 step did not produce {target}")
    print(f"[convert] Wrote: {target}")
    return target


def export_to_tflite(pt_path: Path, imgsz: int) -> Path:
    """Full pipeline PT → ONNX → TFLite FP16, in two explicit stages."""
    onnx_path = export_pt_to_onnx(pt_path, imgsz)
    return export_onnx_to_tflite(onnx_path, imgsz)


def benchmark(pt_path: Path, tflite_path: Path, imgsz: int, runs: int) -> None:
    """Compare inference latency: Ultralytics .pt vs raw TFLite interpreter.

    The two run on different runtimes (PyTorch CPU/GPU vs TFLite CPU), so this is
    not an apples-to-apples kernel comparison — it reflects the real-world latency
    you'd see in each deployment scenario. That's what matters for "is this
    real-time on the target hardware" claims.
    """
    print("\n" + "=" * 60)
    print(f"Benchmark: {runs} iterations at {imgsz}x{imgsz}")
    print("=" * 60)

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # ---- PyTorch / Ultralytics ----
    from ultralytics import YOLO
    pt_model = YOLO(str(pt_path))
    # Warmup
    for _ in range(3):
        pt_model(dummy, verbose=False, imgsz=imgsz)
    t0 = time.perf_counter()
    for _ in range(runs):
        pt_model(dummy, verbose=False, imgsz=imgsz)
    pt_avg_ms = (time.perf_counter() - t0) / runs * 1000

    # ---- TFLite (FP16) ----
    try:
        # Prefer the lighter tflite_runtime if available; fall back to full TF.
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(tflite_path))
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()

        # FP16 TFLite still expects FP32 input — TF auto-casts internally.
        sample = dummy.astype(np.float32)[None] / 255.0
        sample = sample.astype(in_details["dtype"])

        # Warmup
        for _ in range(3):
            interpreter.set_tensor(in_details["index"], sample)
            interpreter.invoke()

        t0 = time.perf_counter()
        for _ in range(runs):
            interpreter.set_tensor(in_details["index"], sample)
            interpreter.invoke()
            for out in out_details:
                interpreter.get_tensor(out["index"])
        tflite_avg_ms = (time.perf_counter() - t0) / runs * 1000
    except Exception as e:
        print(f"[benchmark] TFLite benchmark failed: {e}")
        tflite_avg_ms = None

    # ---- Report ----
    pt_size = pt_path.stat().st_size
    tflite_size = tflite_path.stat().st_size

    print(f"\n{'Format':<20} {'Size':<15} {'Avg latency':<15} {'FPS':<10}")
    print("-" * 60)
    print(f"{'PyTorch (.pt)':<20} {human_size(pt_size):<15} "
          f"{pt_avg_ms:<14.1f}ms {1000/pt_avg_ms:<10.1f}")
    if tflite_avg_ms is not None:
        print(f"{'TFLite FP16':<20} {human_size(tflite_size):<15} "
              f"{tflite_avg_ms:<14.1f}ms {1000/tflite_avg_ms:<10.1f}")
        size_ratio = pt_size / tflite_size
        print(f"\nSize reduction: {size_ratio:.2f}x smaller")
        if tflite_avg_ms < pt_avg_ms:
            print(f"Speedup: {pt_avg_ms / tflite_avg_ms:.2f}x faster")
        else:
            print(f"Note: TFLite slower on this CPU. FP16 TFLite shines on "
                  f"mobile/edge accelerators (Coral, ARM NEON, NPUs); on a "
                  f"desktop CPU PyTorch is often competitive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default 640).")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the PT vs TFLite latency comparison.")
    parser.add_argument("--runs", type=int, default=30,
                        help="Iterations per model in benchmark (default 30).")
    args = parser.parse_args()

    if not PT_MODEL.exists():
        raise FileNotFoundError(
            f"Plate detector not found at {PT_MODEL}.\n"
            f"Download yolo11x.pt from the PlateScanner Yandex Disk and place it there."
        )

    print(f"Source model:   {PT_MODEL} ({human_size(PT_MODEL.stat().st_size)})")
    tflite_path = export_to_tflite(PT_MODEL, args.imgsz)
    print(f"\nExported model: {tflite_path} ({human_size(tflite_path.stat().st_size)})")

    if not args.no_benchmark:
        benchmark(PT_MODEL, tflite_path, args.imgsz, args.runs)


if __name__ == "__main__":
    main()
