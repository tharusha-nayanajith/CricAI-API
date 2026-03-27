import asyncio
from pathlib import Path

import cv2

from app.modules.preprocessor.service import VideoPreprocessor


async def main():
    preprocessor = VideoPreprocessor()
    artifacts = await preprocessor.run(Path("test_video.mp4"))
    cv2.imwrite("release_frame.jpg", artifacts.release_frame)

asyncio.run(main())
