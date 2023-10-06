import socketio
import io, torch
import requests
import json

# Socket.IO 클라이언트 생성
sio = socketio.Client()

# 연결 성공 시 호출
@sio.event
def connect():
    print('Connected to Flask-SocketIO server')

# 서버로부터 message 이벤트를 수신하면 호출
@sio.on('cliRecv')
def clientReceive(json):
    model = torch.jit.load(io.BytesIO(eval(json['model_bytes'])))
    torch.jit.save(model, f"model/model_{json['token']}.pt")
    sio.disconnect()
    exit()

# 서버로부터 토큰 발급
response = requests.post('http://localhost:5000/token', json = {'capa': int(input())})
token = json.loads(response.content)['token']

# Flask-SocketIO 서버 주소로 연결
sio.connect('http://localhost:5000', wait_timeout = 10)
sio.emit('join', token)

# 서버에게 연합 학습 요청 및 모델 전달
requests.post('http://localhost:5000/request', json = {'token': token, 'model_bytes': 'model_bytes'})

# 서버로부터 응답을 받은 후 연결 종료
sio.wait()
# sio.disconnect()
# raise TimeoutError('아직 모든 클라이언트가 연결되지 않았거나 서버가 응답하지 않습니다.')