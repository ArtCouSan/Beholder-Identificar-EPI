python yolov5/train.py --data data.yaml --weights yolov5s.pt --img 640 --epochs 100 --save-period 5

python yolov5/train.py --data data.yaml --weights runs/train/exp8/weights/last.pt --img 640 --epochs 100  --save-period 5

=============================================================================================================================================================

exp11 - 100 epocas (Superajustado)
exp13 - 10 epocas 

=============================================================================================================================================================

! python detect.py --weights runs/train/exp13/weights/best.pt --img 640 --source /content/drive/MyDrive/IC/yolov5/imgs/test/images --data data/data.yaml

=============================================================================================================================================================

sudo apt-get install qt5-default
pip install --upgrade opencv-python opencv-python-headless
