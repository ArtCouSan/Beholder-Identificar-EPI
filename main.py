import torch
import numpy as np
import cv2
import urllib.request
import sys
import os
from flask import Flask, Response
from flask_cors import CORS
import onnxruntime_genai as og

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})

model = torch.hub.load('ultralytics/yolov5', 'custom', "best.pt", force_reload=True)

# Defina o caminho para o modelo Phi-3 Vision e para o genai_config.json
phi_model_path = "./cpu-int4-rtn-block-32-acc-level-4/phi-3-v-128k-instruct-vision.onnx"
genai_config_path = "./cpu-int4-rtn-block-32-acc-level-4/genai_config.json"

# Inicializa o modelo Phi-3 Vision com o caminho do modelo e do genai_config.json
phi_model = og.Model(model_path=phi_model_path, config_path=genai_config_path)
phi_processor = phi_model.create_multimodal_processor()

def detect_bounding_box(frame, conf_threshold=0.8):
    results = model(frame)
    detections = results.pred[0] 

    print("111111111111111111111111111111111111111")
    
    # Processa a imagem com Phi-3 Vision
    images = og.Images.from_array(frame)  # Converte o frame para o formato de imagens suportado pelo Phi-3 Vision
    prompt = "<|user|>\nDetectar se a pessoa está usando capacete corretamente.\n<|end|>\n<|assistant|>\n"
    inputs = phi_processor(prompt, images=images)

    print("2222222222222222222222222222")

    
    params = og.GeneratorParams(phi_model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=7680)

    generator = og.Generator(phi_model, params)
    
    phi_response = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        phi_response += phi_processor.tokenizer.decode(new_token)
    
    # Aqui você pode processar a resposta de Phi-3 Vision e usá-la para ajustar a lógica do YOLOv5
    print(f"Phi-3 Response: {phi_response}")

    # Lógica YOLOv5 para detectar objetos e desenhar bounding boxes
    for *box, conf, cls in detections:
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f'{results.names[int(cls)]} {conf:.2f}'

            # Ajusta a cor baseado na detecção
            if 'capacete' in label and "no braço" in phi_response:
                color = (0, 0, 255)  # Vermelho se o capacete estiver no braço
                label = "capacete no braço"
            elif 'capacete' in label and "na cabeça" in phi_response:
                color = (0, 255, 0)  # Verde se estiver na cabeça
                label = "capacete na cabeça"
            else:
                color = (255, 255, 255)  # Branco para outros casos

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def generate_frames():
    image_url = "http://192.168.15.123:8080/shot.jpg"  # URL para capturar a imagem estática

    while True:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Analisa a imagem com o modelo YOLOv5 e Phi-3 Vision
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
    app.run(debug=True, port=5000)
