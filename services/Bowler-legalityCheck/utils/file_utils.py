import os
from fastapi import UploadFile
import aiofiles
from typing import Tuple

DATA_DIR = "data"
PROF_DIR = os.path.join(DATA_DIR, "professional")
PRACT_DIR = os.path.join(DATA_DIR, "practice")

for d in [DATA_DIR, PROF_DIR, PRACT_DIR]:
    os.makedirs(d, exist_ok=True)

async def save_upload_file(upload_file: UploadFile, save_path: str) -> str:
    """
    Save FastAPI UploadFile to disk asynchronously.
    Returns saved path.
    """
    async with aiofiles.open(save_path, "wb") as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return save_path
