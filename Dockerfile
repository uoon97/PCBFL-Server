FROM bitnami/pytorch:2.0.1

COPY requirements.txt /app/requirements.txt
COPY sioServer.py /app/sioServer.py
COPY utils.py /app/utils.py
COPY methods.py /app/methods.py
COPY container_start.sh /app/container_start.sh
COPY docker_run.sh /app/docker_run.sh

WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "sioServer.py"]