"""
Sample YOLO inference script using the Ultralytics package.

Install:
    pip install ultralytics

Usage:
    python yolo_inference.py --model yolo11n.pt --source path/to/image_or_dir_or_video.mp4
    python yolo_inference.py --model runs/detect/train/weights/best.pt --source 0  # webcam
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_inference(
    model_path: str,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = None,
    save_dir: str = "runs/predict",
    show: bool = False,
):
    """
    Run YOLO inference on an image, directory, video, or webcam stream.

    Args:
        model_path: Path to a .pt weights file (or a model name like 'yolo11n.pt').
        source: Path to image/video/dir, or '0' for webcam.
        conf: Confidence threshold for detections.
        iou: IoU threshold used for NMS.
        device: 'cuda', 'cpu', or None to auto-select.
        save_dir: Directory where annotated results are written.
        show: Whether to display results in a window as they're processed.
    """
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        project=str(Path(save_dir).parent),
        name=Path(save_dir).name,
        show=show,
    )

    for result in results:
        image_path = result.path
        boxes = result.boxes

        print(f"\n{image_path}: {len(boxes)} detection(s)")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            print(
                f"  {cls_name}: conf={confidence:.3f} "
                f"box=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]"
            )

    print(f"\nAnnotated results saved to: {save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference (Ultralytics)")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image/video/dir path, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or leave unset to auto-select")
    parser.add_argument("--save-dir", type=str, default="runs/predict", help="Output directory for results")
    parser.add_argument("--show", action="store_true", help="Display results in a window")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
