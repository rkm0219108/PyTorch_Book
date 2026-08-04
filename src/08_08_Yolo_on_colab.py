# !git clone https://github.com/pjreddie/darknet

# ls -l

# ls -l darknet/

# cd darknet

# ls cfg -l

# !make

# !wget https://pjreddie.com/media/files/yolov3.weights

#!gdown https://pjreddie.com/media/files/yolov3.weights

# ls -l data

# !./darknet detect cfg/yolov3.cfg ./yolov3.weights data/horses.jpg

# mount google drive
from google.colab import (
    drive,
    files,
)

drive.mount('/content/drive')

# 注意 yolov3.weights在 gdrive 所在目錄
# !./darknet detect cfg/yolov3.cfg "/content/drive/My Drive/yolov3.weights" data/dog.jpg


files.upload()
