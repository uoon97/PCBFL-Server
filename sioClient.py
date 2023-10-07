import socketio
import io, torch
import requests
import json

def flAggregation(url, capacity, token = None, model_bytes = int):

    sio = socketio.Client()
    # Socket.IO 클라이언트 생성
    
    # 연결 성공 시 호출
    @sio.event
    def connect():
        print('Connected to Flask-SocketIO server')

    # 서버로부터 message 이벤트를 수신하면 호출
    @sio.on('cliRecv')
    def clientReceive(data):
        print(data)
        sio.disconnect()

    # 서버로부터 토큰 발급
    if token == None:
        response = requests.post(f'{url}:5000/token', json = {'capa': capacity, 'url': url}, verify = False)
        token = json.loads(response.content)['token']
    
    print(token)

    # Flask-SocketIO 서버 주소로 연결
    sio.connect(f'{url}:5000', wait_timeout = 10)
    sio.emit('join', token)

    # 서버에게 연합 학습 요청 및 모델 전달
    requests.post(f'{url}:5000/request', json = {'token': token, 'model_bytes': model_bytes, 'url': url}, verify = False)

    # 서버로부터 응답을 받은 후 연결 종료
    sio.wait()
    # sio.disconnect()
    # raise TimeoutError('아직 모든 클라이언트가 연결되지 않았거나 서버가 응답하지 않습니다.')
    return