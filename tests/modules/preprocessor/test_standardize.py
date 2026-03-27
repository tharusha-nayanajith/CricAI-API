from pathlib import Path

import pytest

from app.exceptions import PreprocessingError
from app.modules.preprocessor.service import VideoPreprocessor


class _FakeCapture:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 3:
            return float(self._width)
        if prop == 4:
            return float(self._height)
        return 0.0

    def release(self) -> None:
        return None


@pytest.mark.asyncio
async def test_standardize_video_returns_original_when_already_standardized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = VideoPreprocessor()
    video_path = Path("C:/tmp/input.mp4")

    monkeypatch.setattr(
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeCapture(720, 1280),
    )

    result = await preprocessor.standardize_video(video_path)

    assert result == video_path


@pytest.mark.asyncio
async def test_standardize_video_calls_ffmpeg_with_expected_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = VideoPreprocessor()
    video_path = Path("C:/tmp/input.mp4")
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeCapture(640, 480),
    )
    monkeypatch.setattr(preprocessor, "_resolve_ffmpeg_binary", lambda: "ffmpeg")

    def _fake_run_ffmpeg(cmd: list[str]):  # noqa: ANN202
        captured["cmd"] = cmd

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr(preprocessor, "_run_ffmpeg", _fake_run_ffmpeg)

    result = await preprocessor.standardize_video(video_path)

    assert result == Path("C:/tmp/input_standardized.mp4")
    assert captured["cmd"][0] == "ffmpeg"
    assert "-vf" in captured["cmd"]
    assert "scale=720:1280" in captured["cmd"]
    assert "-c:v" in captured["cmd"]
    assert "libx264" in captured["cmd"]
    assert "-crf" in captured["cmd"]
    assert "18" in captured["cmd"]
    assert "-preset" in captured["cmd"]
    assert "fast" in captured["cmd"]


@pytest.mark.asyncio
async def test_standardize_video_raises_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = VideoPreprocessor()
    video_path = Path("C:/tmp/input.mp4")

    monkeypatch.setattr(
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeCapture(800, 600),
    )
    monkeypatch.setattr(preprocessor, "_resolve_ffmpeg_binary", lambda: "ffmpeg")

    def _fake_run_ffmpeg(cmd: list[str]):  # noqa: ANN202
        class _Result:
            returncode = 1
            stderr = "ffmpeg failure"

        return _Result()

    monkeypatch.setattr(preprocessor, "_run_ffmpeg", _fake_run_ffmpeg)

    with pytest.raises(PreprocessingError, match="ffmpeg failed"):
        await preprocessor.standardize_video(video_path)
