import cv2
import os
from datetime import datetime

def extract_frames(video_path, output_folder, interval=5):
    """
    Extrai frames de um vídeo MP4 a cada 'interval' segundos e salva como imagens com nomes automáticos
    na mesma pasta.
    
    Args:
    - video_path: Caminho para o vídeo MP4.
    - output_folder: Pasta onde os frames serão salvos.
    - interval: Intervalo de tempo entre os frames a serem capturados, em segundos.
    """
    # Verifica se a pasta de saída existe, caso contrário cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)

    # Obter a taxa de frames por segundo (FPS) do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calcular quantos frames correspondem ao intervalo de tempo (5 segundos)
    frame_interval = int(fps * interval)

    # Verificar se o vídeo foi carregado corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Loop para processar cada frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo ou erro ao ler o frame.")
            break

        # Salvar o frame a cada intervalo de tempo especificado
        if frame_count % frame_interval == 0:
            # Gerar um nome de arquivo único baseado na data e hora atual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_filename = f"{output_folder}/frame_{timestamp}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Frame salvo: {frame_filename}")

        frame_count += 1

    # Liberar o vídeo após o processamento
    cap.release()
    print("Processamento concluído.")

# Exemplo de uso
video_path = "./Thais - sem nada.mp4"
output_folder = "./imagens/"
extract_frames(video_path, output_folder, interval=5)
