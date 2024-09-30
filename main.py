from flask import Flask, Response, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
import urllib.request
import sys
import os
import time
import base64
from pymongo import MongoClient
from datetime import datetime
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

app = Flask(__name__)
CORS(app)

# # Configura a API da OpenAI
# client_open_ai = OpenAI(
#   api_key='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',  # this is also the default, it can be omitted
# )

# Configuração da conexão com MongoDB
client = MongoClient('mongodb://192.168.56.1:27017/')
db = client['epi_database']  # Nome do banco de dados
collection = db['detections']  # Nome da coleção

model = torch.hub.load('ultralytics/yolov5', 'custom', "best.pt", force_reload=True)

last_saved_time = 0  # Armazena o tempo do último salvamento da imagem (em segundos)
segundos = 30

# # Função para validar a detecção com OpenAI
# def validate_with_openai(image_base64, detections):
#     prompt = f"Eu detectei um(a) pessoa {detections[0]['class']} em uma imagem. Confirma que isso é correto? A confiança foi {detections[0]['confidence']}. Sim ou Nao?"
    
#     # Chamada à API OpenAI para validar a detecção com GPT-3.5 Turbo
#     completion = client_open_ai.chat.completions.create(
#         model="gpt-4o-mini",  # Use o GPT-3.5 Turbo para um desempenho mais rápido
#         messages=[
#             {"role": "system", "content": "Você é um assistente que valida detecções de imagens de uso de EPI."},
#             {"role": "user", "content": prompt}
#         ]
#     )
    
#     # Retorna a resposta da OpenAI
#     return completion.choices[0].message.content.strip()

# Função para salvar no MongoDB
def save_image_to_mongodb(frame, detection_data):
    # Codifica a imagem em formato base64 para armazenar no MongoDB
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Cria o documento a ser inserido no MongoDB
    document = {
        'timestamp': datetime.now(),
        'image': image_base64,
        'detections': detection_data
    }

    # Insere o documento no MongoDB
    collection.insert_one(document)
    print(f"Imagem salva no MongoDB com as detecções: {detection_data}")

# Função que realiza a detecção e faz o filtro com a OpenAI
def detect_bounding_box(frame, conf_threshold=0.8):
    global last_saved_time
    results = model(frame)
    detections = results.pred[0]
    detection_list = []
    should_save_image = False

    for *box, conf, cls in detections:
        if conf >= conf_threshold:
            # Desenha o retângulo ao redor do objeto detectado
            x1, y1, x2, y2 = map(int, box)
            label = f'{results.names[int(cls)]} {conf:.2f}'
            detection_data = {
                'class': results.names[int(cls)],
                'confidence': float("{:.2f}".format(conf))
            }
            detection_list.append(detection_data)

            # Se detectar "sem_capacete", "sem_colete" ou "sem_bota", marcamos para salvar
            if label.startswith('sem_capacete') or label.startswith('sem_colete') or label.startswith('sem_bota'):
                should_save_image = True

            # Escolhe a cor do retângulo baseado na classe
            if label.startswith('sem_capacete') or label.startswith('sem_colete') or label.startswith('sem_bota'):
                color = (0, 0, 255)  # Vermelho
            elif label.startswith('capacete') or label.startswith('colete') or label.startswith('bota'):
                color = (0, 255, 0)  # Verde
            elif label.startswith('pessoa'):
                color = (255, 0, 0)  # Azul
            else:
                color = (255, 255, 255)  # Branco para outros casos

            # Desenha o retângulo ao redor do objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Adiciona uma caixa de fundo para o texto
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)  # Caixa de fundo
            # Adiciona o label acima do retângulo
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Texto em preto

    # Verifica se devemos salvar a imagem, com um intervalo de 30 segundos
    current_time = time.time()
    if should_save_image and (current_time - last_saved_time >= segundos):
        # Codifica a imagem em base64 para passar para a OpenAI
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Valida com OpenAI
        # validation_result = validate_with_openai(image_base64, detection_list)
        # print(f"Resultado da validação com OpenAI: {validation_result}")

        # Se a OpenAI confirmar a detecção, salva a imagem no MongoDB
        # if "Sim" in validation_result: 
        save_image_to_mongodb(frame, detection_list)
        last_saved_time = current_time
        # else:
        #     print("OpenAI não confirmou a detecção. A imagem não será salva.")
    
    return frame, detection_list

def generate_frames():
    image_url = "http://192.168.56.1:5001/snapshot"  # URL para capturar a imagem estática

    while True:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Analisa a imagem com o modelo YOLOv5
        frame, _ = detect_bounding_box(frame)  # Ignora as detecções aqui, pois só estamos gerando o vídeo

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

@app.route('/detections')
def detections():
    image_url = "http://192.168.56.1:5001/snapshot"  # URL para capturar a imagem estática

    # Captura a imagem da URL
    img_resp = urllib.request.urlopen(image_url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    # Analisa a imagem com o modelo YOLOv5 e retorna as detecções
    _, detection_list = detect_bounding_box(frame)
    
    # Retorna as detecções como JSON
    return jsonify(detection_list)

# Novo endpoint para retornar as imagens salvas
@app.route('/saved_images', methods=['GET'])
def get_saved_images():
    # Busca todas as imagens salvas no MongoDB
    saved_images = collection.find({}, {"_id": 0, "timestamp": 1, "image": 1, "detections": 1}).sort("timestamp", -1)

    # Prepara a resposta com os dados de timestamp, imagem e detecções
    image_list = []
    for image in saved_images:
        image_list.append({
            'timestamp': image['timestamp'],
            'image': image['image'],  # A imagem está em formato base64
            'detections': image['detections']
        })

    # Retorna a lista de imagens como JSON
    return jsonify(image_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)