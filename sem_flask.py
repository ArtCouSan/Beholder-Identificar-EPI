import cv2
from flask import Flask, Response
from flask_cors import CORS
import torch
import numpy as np
import sys
import os
import urllib

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
from yolov5.models.common import DetectMultiBackend

model = torch.hub.load('ultralytics/yolov5', 'custom', "best.pt", force_reload=True)

image_url = "http://192.168.15.123:8080/shot.jpg"  # URL para capturar a imagem estática


while True:
    img_resp=urllib.request.urlopen(url=image_url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    im = cv2.imdecode(imgnp,-1)

    results = model(im)

    print(results)

    frame = np.squeeze(results.render())

    cv2.imshow('Deteccao', frame)

    key=cv2.waitKey(5)

    if key==ord('q'):
        break