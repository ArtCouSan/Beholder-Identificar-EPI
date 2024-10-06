import vlc
import time

# Configuração do VLC
player = vlc.Instance()

# Definir a mídia como stream RTSP
media = player.media_new("rtsp://admin:beholder2024@192.168.173.64:554/onvif1")

# Adicionar opções para gravar o vídeo no formato MP4, utilizando um codec adequado
media.add_option('sout=#transcode{vcodec=h264,acodec=mp4a}:file{mux=mp4,dst=video_gravado.mp4}')
media.add_option('sout-keep')  # Para manter a gravação ativa

media_player = player.media_player_new()
media_player.set_media(media)
media_player.play()

# Grava por 60 segundos (1 minuto)
time.sleep(60)

# Para a reprodução (e a gravação)
media_player.stop()
