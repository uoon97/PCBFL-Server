import os, io
import pymongo, torch
import random, string
from methods import *

# connect to MongoDB's request collection
def connReq(url):
    client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    db = client["mydatabase"]
    collection = db['requests']
    return collection

def connToken(url):
    client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    db = client["mydatabase"]
    collection = db['tokens']
    return collection

def generateToken(url):
    colReq = connToken(url)
    while True:
        token = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        if colReq.count_documents({'token': token}) == 0:
            return token

def fedModels(token, url):
    colReq = connReq(url)
    documents = colReq.find({"token" : token})
    models = []

    for doc in enumerate(documents):
        models.append(torch.jit.load(io.BytesIO(doc['model_bytes'])))

    # torch.jit.save(fedMethod(models), f'models/model_{token}.pt')

    with open(f"models/model_{token}.pt", "wb") as f:
        model_bytes = f.read()
    
    os.remove(f"models/{token}.pt")
    return model_bytes