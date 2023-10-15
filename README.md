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

    from fedClient import *

    torch.hub.load('ultralytics/yolov5', 'yolov5n')

    with open('model.pt', 'rb') as f:
        model_bytes = f.read()

    token = generate_token(url, capacity = int)

    federate(url, model_bytes = model_bytes, token = token, round = int)

Finally, you can see that a federated model file is created in current directory.

If you can use Docker, try this and Run above python command.

    sh setting_for_client.sh
