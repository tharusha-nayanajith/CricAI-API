"""
Run this ONCE to download the cricket ball dataset from Roboflow.
After downloading, run the training command shown at the bottom.
"""

from roboflow import Roboflow
rf = Roboflow(api_key="fax7YDwW9kRy7YRVsCGD")
project = rf.workspace("kika12").project("cricket-ball-6gtio")
version = project.version(1)
dataset = version.download("yolov8")
                

print("\n✅ Dataset downloaded!")
print(f"📁 Dataset location: {dataset.location}")
print("\nNow run training with:")
print(f"  yolo detect train data={dataset.location}/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8")
