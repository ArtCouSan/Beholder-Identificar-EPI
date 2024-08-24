from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
from roboflow import Roboflow

# Configuração do Flask
app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})

# Inicialização do modelo da Roboflow
rf = Roboflow(api_key="rAqfJZfIrCve7AXjKcc4")  
project = rf.workspace().project("construction-site-safety")
model = project.version(27).model

def detect_bounding_box(frame):
    # Salva o frame atual como um arquivo temporário
    cv2.imwrite("temp_frame.jpg", frame)
    
    # Realiza a predição usando o modelo da Roboflow
    prediction = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()

    # Itera sobre as predições para desenhar as bounding boxes
    for pred in prediction['predictions']:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        w = int(pred['width'])
        h = int(pred['height'])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, pred['class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            detect_bounding_box(frame)
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
