FROM bitnami/pytorch:2.0.1

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]