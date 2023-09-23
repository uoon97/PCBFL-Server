import os
import random, string
import pymysql, pymongo
import torch
from methods import *

# connect to RDS
def rdb_connect():
    host, user, password, db = db_info()

    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset='utf8',
        port = 3306
    )

    return connection

# generate unique token with 16 characters
def generate_token():

    while True:

        characters = string.ascii_letters + string.digits + string.punctuation
        random_string = ''.join(random.choice(characters) for i in range(16))


        connection = rdb_connect()
        selectQuery = f"SELECT * FROM tokens WHERE token = '{random_string}'"
    
        with connection.cursor() as cursor:
            cursor.execute(selectQuery)
            result = cursor.fetchone()
        
        if result is None:
            return random_string

# insert rdb into token's info
def token2rdb(token, user_id, capa, method):
    connection = rdb_connect()
    insertQuery = f"INSERT INTO tokens VALUES ('{token}', '{user_id}', '{capa}', '{method}')"

    with connection.cursor() as cursor:
        cursor.execute(insertQuery)

    connection.commit()
    connection.close()

# view token's capa
def rdb2capa(token):
    connection = rdb_connect()
    selectQuery = f"SELECT capa FROM tokens WHERE token = '{token}'"

    with connection.cursor() as cursor:
        cursor.execute(selectQuery)
        capa = cursor.fetchone()[0]

    connection.close()

    return capa

# view token's capa
def rdb2method(token):
    connection = rdb_connect()
    selectQuery = f"SELECT method FROM tokens WHERE token = '{token}'"

    with connection.cursor() as cursor:
        cursor.execute(selectQuery)
        method = cursor.fetchone()[0]

    connection.close()

    return method

# whether token is valid or not by searching token in rdb
def rdb2Auth(token):
    connection = rdb_connect()
    selectQuery = f"SELECT * FROM tokens WHERE token = '{token}'"
    
    with connection.cursor() as cursor:
        cursor.execute(selectQuery)
        result = cursor.fetchone()

    connection.close()

    if result is None:
        return False
    return True

# connect to MongoDB
def ndb_connect():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["mydatabase"]
    return db
    # collection = db["posts"]
    # return collection

def ndb2userTokens(username):
    db = ndb_connect()
    collection = db["users"]
    document = collection.find({"username" : username})
    tokens = []
    for doc in document:
        tokens.append(doc['token'])
    return tokens

def ndb2userModels(tokens):
    db = ndb_connect()
    collection = db["models"]
    models = []
    for token in tokens:
        document = collection.find({"token" : token})
        models.append(document[0]['model'])
    return models

def aggregate_model(token):
    # collect models
    # posts = (username, token, model)
    # users = (username, token)
    db = ndb_connect()
    collection = db["posts"]
    document = collection.find({"token" : token})
    models = []

    collection = db["users"]
    for doc in document:
        collection.insert_one({"username": doc['username'], "token": doc['token']})
        with open(f"models/{doc['username']}_{doc['token']}.pt", "wb") as f:
            f.write(doc['model'])
        models.append(torch.jit.load(f"models/{doc['username']}_{doc['token']}.pt"))
        os.remove(f"models/{doc['username']}_{doc['token']}.pt")

    # aggregate models
    method = rdb2method(token)

    # with open(f"models/{doc['token']}.pt", "wb") as f:
    #     f.write(fl_model)

    # save aggregated model
    # models = (token, model)
    db = ndb_connect()
    collection = db["models"]

    # with open(f"models/{doc['token']}.pt", "rb") as f:
    #     fl_model = f.read()
    #     os.remove(f"models/{doc['token']}.pt")

    # data = {"token" : token, "model" : fl_model}
    # collection.insert_one(data)

    return 