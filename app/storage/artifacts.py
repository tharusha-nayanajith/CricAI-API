from __future__ import annotations

import os
from pathlib import Path

ARTIFACTS_ROOT = Path(os.getenv("CRICKAI_ARTIFACTS_DIR", "/tmp/crickai-artifacts"))


def _feature_dir(job_id: str, feature_name: str) -> Path:
    return (ARTIFACTS_ROOT / job_id / feature_name).resolve()


def get_artifact_path(job_id: str, feature_name: str, artifact_name: str) -> Path:
    base_dir = _feature_dir(job_id, feature_name)
    candidate = (base_dir / Path(artifact_name).name).resolve()
    if base_dir not in candidate.parents and candidate != base_dir:
        raise ValueError("Invalid artifact path")
    return candidate


def build_artifact_url(job_id: str, feature_name: str, artifact_name: str) -> str:
    safe_name = Path(artifact_name).name
    return f"/results/{job_id}/artifacts/{feature_name}/{safe_name}"


def write_image_artifact(job_id: str, feature_name: str, artifact_name: str, image_bgr) -> str | None:
    import cv2

    output_path = get_artifact_path(job_id, feature_name, artifact_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        return None
    return build_artifact_url(job_id, feature_name, artifact_name)
