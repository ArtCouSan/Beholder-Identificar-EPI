from flask import Flask, Response
from flask_cors import CORS
import torch
import numpy as np
import cv2
import urllib.request
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
from yolov5.models.common import DetectMultiBackend

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})

model = torch.hub.load('ultralytics/yolov5', 'custom', "best.pt", force_reload=True)

def detect_bounding_box(frame, conf_threshold=0.6):
    results = model(frame)

    # Filtra as detecções com base no conf_threshold
    detections = results.pred[0]  # pred[0] contém as detecções da primeira imagem

    for *box, conf, cls in detections:
        if conf >= conf_threshold:
            # Desenha o retângulo ao redor do objeto detectado
            x1, y1, x2, y2 = map(int, box)
            label = f'{results.names[int(cls)]} {conf:.2f}'

            # Desenha o retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Adiciona o label acima do retângulo
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def generate_frames():
    image_url = "http://192.168.15.123:8080/shot.jpg"  # URL para capturar a imagem estática

    while True:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Analisa a imagem com o modelo YOLOv5
        frame = detect_bounding_box(frame)

        # Codifica a imagem processada de volta para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)