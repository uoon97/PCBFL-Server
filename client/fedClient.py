from fake_useragent import UserAgent
import socketio
import io, torch
import requests
import json

def generate_token(url, capacity):
    response = requests.post(f'{url}:5000/token', json = {'capa': capacity, 'url': url}, verify = False)
    token = json.loads(response.content)['token']
    return token


def federate(url, model_bytes, token, round = int):
    # Socket.IO 클라이언트 생성
    sio = socketio.Client()

    # 연결 성공 시 호출
    @sio.event
    def connect():
        print('Connected to Flask-SocketIO server')

    # 서버로부터 message 이벤트를 수신하면 호출
    @sio.on('cliRecv')
    def clientReceive(json):
        torch.hub.load('ultralytics/yolov5', 'yolov5n')
        model = torch.load(io.BytesIO(eval(json['model_bytes'])))
        torch.save(model, f"model{round}_{json['token']}.pt")
        sio.disconnect()

    # Flask-SocketIO 서버 주소로 연결
    sio.connect(f'{url}:5000', wait_timeout = 10)
    sio.emit('join', token)

    # 서버에게 연합 학습 요청 및 모델 전달
    user_agent = UserAgent()
    headers = {'User-Agent': user_agent.random}
    requests.post(f'{url}:5000/request', json = {'token': token, 'model_bytes': str(model_bytes), 'url': url}, headers = headers, verify = False)

    # 서버로부터 응답을 받은 후 연결 종료
    sio.wait()
