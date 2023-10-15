import pymongo
import random, string

# connect to MongoDB's request collection
def connReq(url):
    client = pymongo.MongoClient(f"mongodb://{url.split('//')[1]}:27017/")
    db = client["mydatabase"]
    collection = db['requests']
    return collection

def connToken(url):
    client = pymongo.MongoClient(f"mongodb://{url.split('//')[1]}:27017/")
    db = client["mydatabase"]
    collection = db['tokens']
    return collection

def generateToken(url):
    colReq = connToken(url)
    while True:
        token = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        if colReq.count_documents({'token': token}) == 0:
            return token