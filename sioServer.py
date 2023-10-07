from flask import Flask, request
from flask_socketio import SocketIO, join_room
from utils import *

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('join')
def onJoin(token):
    join_room(token)

# token room의 subscriber(client)에게 FL_model 전달.
@socketio.on('servSend')
def serverSend(token, url):
    colToken = connToken(url)
    colReq = connReq(url)
    documents = colReq.find({'token': token})

    mean = 0
    for doc in documents:
        mean += doc['model_bytes']
    mean /= colReq.count_documents({'token': token})

    socketio.emit('cliRecv', mean, room = token)
    
    colReq.delete_many({'token': token})
    colToken.delete_one({'token': token})


# token 생성 및 token의 capa 저장.
@app.route('/token', methods = ['POST'])
def genToken():
    if request.method == 'POST':
        # insert token to MongoDB
        json = request.get_json()
        token = generateToken(json['url'])
        colToken = connToken(json['url'])
        colToken.insert_one({'token': token, 'capa': json['capa']})
        return {'token': token}
    

# client로부터 모델 수신 및 capa 충족 시 연합 학습과 클라이언트로 모델 전달. 
@app.route('/request', methods = ['POST'])
def cliReq():
    if request.method == 'POST':
        # json = {'token': token, 'model_bytes': model_bytes}
        json = request.get_json()
        
        colReq = connReq(json['url'])
        colReq.insert_one(json)

        colToken = connToken(json['url'])
        capacity = colToken.find_one({'token': json['token']})['capa']

        if colReq.count_documents({'token': json['token']}) == capacity:
            serverSend(json['token'], json['url'])

        return {'status': 'success'}

if __name__ == '__main__':
    socketio.run(app, host = '0.0.0.0', port=5000, allow_unsafe_werkzeug = True)
