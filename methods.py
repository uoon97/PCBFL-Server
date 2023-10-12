# Federated Aggreagation Methods

import torch, tensorflow as tf
from utils import *
import copy, io
import torch

class federation:

    def __init__(self, token, url, method = None):

        self.token = token

        colReq = connReq(url)
        documents = colReq.find({"token" : token})
        models = [torch.load(io.BytesIO(eval(doc['model_bytes'])))['model'] for doc in documents]

        self.models = models

        if method is not None:
            if method == 'fedavg':
                return self.fedavg()
            
            if method == 'fedprox':
                return self.fedprox()

    def to_bytes(self, model):
        torch.jit.save(model, f"models/{self.token}.pt")

        with open(f"models/{self.token}.pt", "wb") as f:
            model_bytes = f.read()

        os.remove(f"models/{self.token}.pt")
        return model_bytes


    def fedavg(self):
        model = torch.load('yolov5m.pt')
        fed_dict = {}
        for key in model.state_dict().keys():
            fed_dict[key] = sum([m.state_dict()[key] for m in self.models])/len(self.models)

        model.load_state_dict(fed_dict)
        return self.to_bytes(model)
        


