import vlc
import cv2
from flask import Flask, Response

app = Flask(__name__)

# Configuração do VLC
player = vlc.Instance()
media = player.media_new("rtsp://admin:beholder2024@192.168.173.64:554/onvif1")
media_player = player.media_player_new()
media_player.set_media(media)
media_player.play()

def capture_frame():
    # Captura um frame via VLC
    frame = media_player.video_take_snapshot(0, 'snapshot.jpg', 0, 0)
    if frame == -1:
        print("Falha ao capturar frame via VLC.")
        return None

    # Lê a imagem capturada
    img = cv2.imread('snapshot.jpg')
    if img is None:
        print("Erro ao carregar a imagem.")
        return None

    # Codifica a imagem como JPEG
    ret, buffer = cv2.imencode('.jpg', img)
    if not ret:
        print("Erro ao codificar a imagem.")
        return None

    return buffer.tobytes()

@app.route('/snapshot')
def snapshot():
    frame_bytes = capture_frame()
    if frame_bytes is None:
        return "Erro ao capturar a imagem.", 500

    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
