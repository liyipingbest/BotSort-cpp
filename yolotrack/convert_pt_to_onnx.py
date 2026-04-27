#!/usr/bin/env python3
"""
Convert YOLO .pt models to OpenCV DNN compatible ONNX format.

The standard ultralytics ONNX export outputs shape [1, 4+C, N], but the
YOLODetector postprocess in this project expects [1, N, 4+C]. This script
inserts a Transpose node to fix the output layout.

Usage:
    python3 convert_pt_to_onnx.py models/yolo11n.pt
    python3 convert_pt_to_onnx.py --all
    python3 convert_pt_to_onnx.py models/yolo11m.pt --output models/yolo11m.onnx

Requirements:
    pip install ultralytics onnx onnxsim
"""

import argparse
import sys
from pathlib import Path


def check_deps():
    """Raise a clear error if required packages are missing."""
    missing = []
    for mod, pkg in [("ultralytics", "ultralytics"),
                     ("onnx", "onnx"),
                     ("onnxsim", "onnxsim")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            "Missing packages: {}\n"
            "Install with: pip install {}".format(
                " ".join(missing), " ".join(missing)
            )
        )


def add_transpose_node(onnx_model, perm=(0, 2, 1)):
    """
    Insert a Transpose node after the last graph output to convert layout
    from [batch, 4+C, N] to [batch, N, 4+C].
    """
    import onnx
    from onnx import helper, TensorProto

    graph = onnx_model.graph
    output = graph.output[0]

    perm_init = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{output.name}_perm"],
        value=helper.make_tensor(
            name=f"{output.name}_perm_value",
            data_type=TensorProto.INT64,
            dims=(3,),
            vals=list(perm),
        ),
    )
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[output.name, f"{output.name}_perm"],
        outputs=[f"{output.name}_transposed"],
    )

    shape = output.type.tensor_type.shape
    new_output = helper.make_tensor_value_info(
        f"{output.name}_transposed",
        output.type.tensor_type.elem_type,
        [shape.dim[p] for p in perm],
    )

    graph.node.extend([perm_init, transpose_node])
    graph.output.pop()
    graph.output.append(new_output)

    onnx.checker.check_model(onnx_model)
    return onnx_model


def export_pt_to_onnx(pt_path, output_path=None, imgsz=640, opset=12):
    """Export a .pt YOLO model to ONNX with output shape [1, N, 4+C]."""
    import onnx
    from ultralytics import YOLO

    model = YOLO(str(pt_path))

    # Standard ultralytics export produces [1, 4+C, N]
    use_simplify = True
    try:
        tmp_path = model.export(
            format="onnx", imgsz=imgsz, opset=opset,
            simplify=True, device="cpu",
        )
    except Exception:
        # onnxsim may not be available; retry without simplify
        print("  (onnxsim unavailable, exporting without simplify)")
        use_simplify = False
        tmp_path = model.export(
            format="onnx", imgsz=imgsz, opset=opset,
            simplify=False, device="cpu",
        )

    tmp_path = Path(tmp_path)
    print(f"  Standard ONNX exported: {tmp_path}")

    # Insert Transpose node
    onnx_model = onnx.load(str(tmp_path))
    onnx_model = add_transpose_node(onnx_model)

    # Optional: simplify again after adding Transpose
    if use_simplify:
        try:
            import onnxsim
            onnx_model, _ = onnxsim.simplify(onnx_model)
        except Exception:
            pass

    out_path = output_path or pt_path.with_suffix(".onnx")
    onnx.save(onnx_model, str(out_path))

    # Remove intermediate file
    if tmp_path != out_path and tmp_path.exists():
        tmp_path.unlink()

    print(f"  Final ONNX saved: {out_path}")
    return out_path


def main():
    check_deps()

    parser = argparse.ArgumentParser(
        description="Convert YOLO .pt models to OpenCV DNN compatible ONNX"
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
        help="ONNX opset version for OpenCV DNN (default: 12)",
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
