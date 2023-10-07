FROM bitnami/pytorch:2.0.1

COPY requirements.txt /app/requirements.txt
COPY sioServer.py /app/sioServer.py
COPY utils.py /app/utils.py
COPY methods.py /app/methods.py

WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "sioServer.py"]