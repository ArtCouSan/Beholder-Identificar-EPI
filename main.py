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
import requests
from roboflow import Roboflow
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

ip = "192.168.56.1"

telegram_token = "7696933448:AAHRSUGvQDgp_58Lte8v0POTemTDtjuiS4g"
chat_id = 937251721  # Seu chat_id do Telegram

# Informações da API do Roboflow
roboflow_api_url = "https://detect.roboflow.com/2?api_key="
# Inicializa o Roboflow
rf = Roboflow(api_key="AtnJ1u3RbFfoGCrenWYH")
project = rf.workspace().project("epi-fiap")  # Nome do seu projeto no Roboflow
model = project.version(2).model  # Número da versão do modelo

app = Flask(__name__)
CORS(app)

# Configuração da conexão com MongoDB
client = MongoClient(f'mongodb://{ip}:27017/')
db = client['epi_database']  # Nome do banco de dados
collection = db['detections']  # Nome da coleção

# model = torch.hub.load('ultralytics/yolov5', 'custom', "best.pt", force_reload=True)

segundos = 30

# Variável global para armazenar o tempo do último envio ao Telegram
last_telegram_time = 0  # Armazena o tempo do último envio ao Telegram (em segundos)

# Função para enviar uma mensagem e imagem ao Telegram via HTTP request
def send_telegram_alert(detection_list, image_path):
    """
    Envia uma mensagem ao Telegram sobre o que foi detectado e anexa uma imagem com a legenda.
    """
    global last_telegram_time  # Acessa a variável global
    current_time = time.monotonic() 

    # Só envia se o intervalo de 30 segundos tiver passado
    if current_time - last_telegram_time >= segundos:
        for detection in detection_list:
            detectado = False
            if detection['class'] == 'sem_capacete':
                message = "⚠️ Alerta! Funcionário identificado **sem capacete**."
                detectado = True
            elif detection['class'] == 'sem_colete':
                message = "⚠️ Alerta! Funcionário identificado **sem colete**."
                detectado = True
            else:
                message = f"Funcionário detectado com **{detection['class']}**. Confiança: {detection['confidence']:.2f}"
                detectado = False

            # Se a detecção for relevante, enviar a imagem com a legenda
            if detectado:
                url_photo = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    payload_photo = {
                        "chat_id": chat_id,
                        "caption": message,  # A legenda é enviada com a imagem
                        "parse_mode": "Markdown"  # Para suportar o formato Markdown na legenda
                    }
                    files = {
                        'photo': photo
                    }
                    try:
                        response_photo = requests.post(url_photo, data=payload_photo, files=files)
                        if response_photo.status_code == 200:
                            print("Imagem e mensagem enviadas com sucesso!")
                            last_telegram_time = current_time  # Atualiza o tempo do último envio
                        else:
                            print(f"Falha ao enviar imagem e mensagem. Status Code: {response_photo.status_code}, Response: {response_photo.text}")
                    except Exception as e:
                        print(f"Erro ao enviar imagem e mensagem ao Telegram: {e}")
    else:
        print(f"Esperando {segundos} segundos para enviar nova imagem ao Telegram.")


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

    # Passa o caminho da imagem para a função `send_telegram_alert`
    send_telegram_alert(detection_data, "temp_snapshot_with_boxes.jpg")

# Variáveis globais para armazenar o último frame e as detecções
last_frame = None
last_detections = None
last_saved_time = 0  # Armazena o tempo do último salvamento da imagem (em segundos)

def detect_bounding_box(frame, conf_threshold=0.75):
    global last_saved_time, last_frame, last_detections, last_telegram_time  # Tornar essas variáveis acessíveis
    
    # Arquivos temporários para as imagens
    temp_image_path_original = "temp_snapshot_original.jpg"
    temp_image_path_with_boxes = "temp_snapshot_with_boxes.jpg"
    
    # Salva o frame original em um arquivo temporário para referência
    cv2.imwrite(temp_image_path_original, frame)
    
    # Faz a inferência usando o modelo do Roboflow
    result = model.predict(temp_image_path_original, confidence=conf_threshold * 100, overlap=30).json()
    
    detection_list = []  # Inicializa corretamente a lista de detecções
    
    should_save_image = False

    # Processa as detecções e desenha as bordas
    for detection in result['predictions']:
        if 'x' in detection and 'y' in detection and 'width' in detection and 'height' in detection:
            x1 = int(detection['x'] - detection['width'] / 2)
            y1 = int(detection['y'] - detection['height'] / 2)
            x2 = int(detection['x'] + detection['width'] / 2)
            y2 = int(detection['y'] + detection['height'] / 2)
            label = f"{detection['class']} {detection['confidence']:.2f}"
            
            detection_data = {
                'class': detection['class'],
                'confidence': float("{:.2f}".format(detection['confidence']))
            }
            
            # Adiciona a detecção à lista de detecções
            detection_list.append(detection_data)

            # Se detectar "sem_capacete", "sem_colete" ou "sem_bota", marcamos para salvar
            if detection['class'] == 'sem_capacete':
                should_save_image = True

            # Escolhe a cor do retângulo baseado na classe
            if detection['class'] == 'sem_capacete' or detection['class'] == 'sem_colete':
                color = (0, 0, 255)  # Vermelho
            elif detection['class'] == 'capacete' or detection['class'] == 'colete':
                color = (0, 255, 0)  # Verde
            elif detection['class'] == 'pessoa':
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

    # Salva a imagem processada com as bordas em um arquivo separado
    cv2.imwrite(temp_image_path_with_boxes, frame)

    # Atualiza os valores globais com o último frame e as detecções
    last_frame = frame
    last_detections = detection_list

    # Verifica se devemos salvar a imagem e enviar para o Telegram, com um intervalo de 30 segundos
    current_time = time.monotonic()
    if should_save_image and (current_time - last_saved_time >= segundos):
        # Iniciar uma nova thread para o processamento do MongoDB e do Telegram
        thread = threading.Thread(target=process_detection_in_thread, args=(frame, detection_list, temp_image_path_with_boxes))
        thread.start()
        # Atualiza o tempo do último salvamento no MongoDB
        last_saved_time = current_time
    
    return frame, detection_list


def process_detection_in_thread(frame, detection_list, image_with_boxes_path):
    # Salva a imagem e as detecções no MongoDB
    save_image_to_mongodb(frame, detection_list)
    
    # Envia a imagem com as bordas e a mensagem para o Telegram
    send_telegram_alert(detection_list, image_with_boxes_path)


def generate_frames():
    image_url = f"http://{ip}:5001/snapshot"  # URL para capturar a imagem estática

    while True:
        # Captura a imagem da URL
        img_resp = urllib.request.urlopen(image_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # Analisa a imagem com o modelo Roboflow e armazena o último frame e suas detecções
        frame, _ = detect_bounding_box(frame)  # Realiza a detecção com o Roboflow

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
    global last_detections
    if last_detections:
        # Retorna as últimas detecções salvas
        return jsonify(last_detections)
    else:
        return jsonify({"message": "Nenhuma detecção disponível."}), 404


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