#!/usr/bin/env python3
"""
Convert YOLO .pt models to ONNX format for OpenCV DNN.

Usage:
    python3 convert_pt_to_onnx.py models/yolo11n.pt
    python3 convert_pt_to_onnx.py --all
    python3 convert_pt_to_onnx.py models/yolo11m.pt --output models/yolo11m.onnx

Requirements:
    pip install ultralytics onnx
"""

import argparse
import sys
from pathlib import Path


def check_deps():
    missing = []
    for mod in ("ultralytics",):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(
            f"Missing packages: {' '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


def export_pt_to_onnx(pt_path, output_path=None, imgsz=640, opset=12):
    """Export a .pt YOLO model to ONNX."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    model.export(
        format="onnx", imgsz=imgsz, opset=opset,
        simplify=True, device="cpu",
    )

    # ultralytics saves next to the source .pt with .onnx extension
    default_path = pt_path.with_suffix(".onnx")

    if output_path and Path(output_path) != default_path:
        Path(default_path).rename(output_path)
        print(f"Saved: {output_path}")
    else:
        print(f"Saved: {default_path}")


def main():
    check_deps()

    parser = argparse.ArgumentParser(
        description="Convert YOLO .pt models to ONNX"
    )
    parser.add_argument(
        "model", nargs="?", default=None,
        help="Path to .pt model file",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all .pt files in models/ directory",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .onnx path (single model only)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset version (default: 12)",
    )
    args = parser.parse_args()

    models_dir = Path(__file__).resolve().parent / "models"

    if args.all:
        pt_files = sorted(models_dir.glob("*.pt"))
        if not pt_files:
            sys.exit(f"No .pt files found in {models_dir}")
        print(f"Found {len(pt_files)} model(s):")
        for pt in pt_files:
            print(f"  - {pt.name}")
        for pt in pt_files:
            print(f"\nConverting {pt.name}...")
            export_pt_to_onnx(pt, imgsz=args.imgsz, opset=args.opset)
        print("\nAll done.")
    elif args.model:
        pt_path = Path(args.model)
        if not pt_path.exists():
            sys.exit(f"Model not found: {pt_path}")
        if pt_path.suffix != ".pt":
            sys.exit(f"Expected .pt file, got: {pt_path}")
        print(f"Converting {pt_path.name}...")
        export_pt_to_onnx(pt_path, output_path=args.output,
                          imgsz=args.imgsz, opset=args.opset)
        print("Done.")
    else:
        pt_files = sorted(models_dir.glob("*.pt"))
        if pt_files:
            print(f"Found {len(pt_files)} model(s) in models/:")
            for pt in pt_files:
                print(f"  - {pt.name}")
            for pt in pt_files:
                print(f"\nConverting {pt.name}...")
                export_pt_to_onnx(pt, imgsz=args.imgsz, opset=args.opset)
            print("\nAll done.")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
