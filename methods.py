# Federated Aggreagation Methods
import copy
import torch
from util import *
import io, os

torch.hub.load('ultralytics/yolov5', 'yolov5m')

class federation:

    def __init__(self, token, url, method = None):

        self.token = token

        colReq = connReq(url)
        documents = colReq.find({"token" : token})
        models = [torch.load(io.BytesIO(eval(doc['model_bytes']))) for doc in documents]
        self.models = models

    def to_bytes(self, model):
        torch.jit.save(model, f"models/{self.token}.pt")

        with open(f"models/{self.token}.pt", "wb") as f:
            model_bytes = f.read()

        os.remove(f"models/{self.token}.pt")
        return model_bytes


    def fedavg(self):
        print('fedavg')
        model = copy.deepcopy(self.models[0])
        fed_dict = {}
        for key in model['model'].state_dict().keys():
            fed_dict[key] = sum([m['model'].state_dict()[key] for m in self.models])/len(self.models)

        model['model'].load_state_dict(fed_dict)
        return self.to_bytes(model)
        


