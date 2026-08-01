# Download YOLOv7 repository and install requirements
# !git clone https://github.com/WongKinYiu/yolov7.git
# %cd yolov7
# !pip install -r requirements.txt

# https://public.roboflow.com/object-detection/aquarium/2

# from google.colab import files
# files.upload()

from google.colab import drive

drive.mount('/content/drive')

# ls /content/drive/MyDrive/0/OID.*

# !unzip /content/drive/MyDrive/0/OID.zip -d .

# download COCO starting checkpoint
# %cd /content/yolov7
# !gdown "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"

import datetime

start = datetime.datetime.now()
start

# run this cell to begin training
# !python train.py --batch 4 --cfg cfg/training/yolov7.yaml --img 640 --epochs 55 --data ./OID/Dataset/data.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml --device 0

print(datetime.datetime.now())
print((datetime.datetime.now() - start).total_seconds())

print((datetime.datetime.now() - start).total_seconds() / 60)

# ls ./runs/train/

# ls ./runs/train/yolov7/weights/

from google.colab import files

files.download('./runs/train/yolov73/weights/best.pt')

# Run evaluation
# !python detect.py --weights ./runs/train/yolov73/weights/best.pt --conf 0.03 --source ./OID/Dataset/test/Balloon/76e41712939b97f2.jpg


# ls /content/yolov7/runs/detect/

from IPython.display import Image, display

imageName = './OID/Dataset/test/Balloon/76e41712939b97f2.jpg'
display(Image(filename=imageName))

# display inference on ALL test images

import glob
from IPython.display import Image, display

i = 0
limit = 10000  # max images to print
for imageName in glob.glob('/content/yolov7/runs/detect/exp2/*.jpg'):  # assuming JPG
    if i < limit:
        display(Image(filename=imageName))
        print("\n")
    i = i + 1
