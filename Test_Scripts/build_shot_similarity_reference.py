from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.modules.action_legality.service import _extract_keypoints_from_frame
from app.modules.shot_similarity.models import PoseKeypoint, ShotReference

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_LIBRARY_PATH = (
    BASE_DIR.parent / "app" / "modules" / "shot_similarity" / "assets" / "golden_frames.json"
)


def load_reference_library() -> dict[str, dict[str, dict[str, object]]]:
    if not REFERENCE_LIBRARY_PATH.exists():
        return {}
    return json.loads(REFERENCE_LIBRARY_PATH.read_text(encoding="utf-8"))


def save_reference_library(payload: dict[str, dict[str, dict[str, object]]]) -> None:
    REFERENCE_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_LIBRARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_keypoints_from_image(image_path: Path) -> list[PoseKeypoint]:
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    keypoints_array = _extract_keypoints_from_frame(frame_bgr, list(range(33)))
    if keypoints_array is None:
        raise ValueError("No pose detected in image. Make sure the full body is visible.")

    keypoints: list[PoseKeypoint] = []
    for idx in range(0, len(keypoints_array), 3):
        keypoints.append(
            PoseKeypoint(
                x=float(keypoints_array[idx]),
                y=float(keypoints_array[idx + 1]),
                z=float(keypoints_array[idx + 2]),
                visibility=1.0,
            )
        )
    return keypoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a shot reference to golden_frames.json.")
    parser.add_argument("--image", required=True, help="Path to the reference shot image.")
    parser.add_argument("--player", required=True, help="Player name, for example 'Virat Kohli'.")
    parser.add_argument("--shot", required=True, help="Shot type, for example 'cover_drive'.")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    keypoints = extract_keypoints_from_image(image_path)
    reference_library = load_reference_library()
    reference_library.setdefault(args.player, {})
    reference_library[args.player][args.shot] = ShotReference(
        keypoints=keypoints
    ).model_dump()
    save_reference_library(reference_library)

    print(
        json.dumps(
            {
                "reference_library": str(REFERENCE_LIBRARY_PATH),
                "player": args.player,
                "shot": args.shot,
                "keypoints": len(keypoints),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
