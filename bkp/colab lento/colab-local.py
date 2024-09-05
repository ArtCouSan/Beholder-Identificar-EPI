from flask import Flask, Response
from flask_cors import CORS
import requests
import cv2
import numpy as np
import urllib.request

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})

# URL da API no Colab (obtenha o endereço da instância do Colab, que pode ser algo como https://<instance_id>.ngrok.io/process_image)
colab_api_url = "https://e945-34-168-11-3.ngrok-free.app/process_image"

def generate_frames():
    image_url = "http://192.168.15.123:8080/shot.jpg"  # URL para capturar a imagem estática

    while True:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Envia a imagem para a API do Colab para processamento
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(colab_api_url, files={'image': img_encoded.tobytes()})

        # Recebe a imagem processada da API do Colab
        img_np = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

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
    app.run(debug=True, port=5000)