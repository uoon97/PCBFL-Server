# PCB_FL

communication server for Federated Object Detection, Image Classification using Flask, python-socketIO.

## How to

### Server

Clone git repo on EC2:

    sudo yum update -y
    sudo yum install git -y
    git clone https://github.com/uoon97/PCB_FL

Run shell script files to build docker image:

    cd PCB_FL

    sh docker_start.sh
    sh container_run.sh

### Client

capacity: number of users to participate in FL
token: By setting the token to None, the token is created, and this token is shared with those who want to participate in the FL.
model_bytes: Reading the model file as a bytes object via file.read()

    from sioClient import *

    with open('model.pt', 'rb') as f:
        model_bytes = f.read()

    flAggregation(url, capacity = capacity, token = None, model_bytes = model_bytes)

Finally, you can see that a federated model file is created in the model directory.
