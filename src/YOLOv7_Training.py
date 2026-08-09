# Download YOLOv7 repository and install requirements
# !git clone https://github.com/WongKinYiu/yolov7.git
# %cd yolov7
# !pip install -r requirements.txt

# https://public.roboflow.com/object-detection/aquarium/2

# from google.colab import files
# files.upload()

import datetime

from google.colab import (
    drive,
    files,
)

drive.mount('/content/drive')

# ls /content/drive/MyDrive/0/OID.*

# !unzip /content/drive/MyDrive/0/OID.zip -d .

# download COCO starting checkpoint
# %cd /content/yolov7
# !gdown "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"


start = datetime.datetime.now()
start

# run this cell to begin training
# !python train.py --batch 4 --cfg cfg/training/yolov7.yaml --img 640 --epochs 55 --data OID/Dataset/data.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml --device 0

print(datetime.datetime.now())
print((datetime.datetime.now() - start).total_seconds())

print((datetime.datetime.now() - start).total_seconds() / 60)

# ls runs/train/

# ls runs/train/yolov7/weights/


files.download('runs/train/yolov73/weights/best.pt')

# Run evaluation
# !python detect.py --weights runs/train/yolov73/weights/best.pt --conf 0.03 --source OID/Dataset/test/Balloon/76e41712939b97f2.jpg
