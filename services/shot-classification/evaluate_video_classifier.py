"""Root launcher for EfficientNetB4+GRU evaluation script.

python.exe evaluate_video_classifier.py --dataset datasets/cricket-shots 
 --model-dir features/SHOT_CLASSIFICATION_SYSTEM/trained_models 
 --shots cut,drive,flick,misc,pull,slog,sweep
"""

import os
import runpy


if __name__ == "__main__":
    target = os.path.join(
        os.path.dirname(__file__),
        "features",
        "SHOT_CLASSIFICATION_SYSTEM",
        "evaluate_video_classifier.py",
    )
    runpy.run_path(target, run_name="__main__")