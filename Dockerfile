FROM bitnami/pytorch:2.0.1

COPY requirements.txt /app/requirements.txt
COPY fedServer.py /app/fedServer.py
COPY utils.py /app/utils.py
COPY methods.py /app/methods.py
COPY yolov5m.pt /app/yolov5m.pt

WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "fedServer.py"]