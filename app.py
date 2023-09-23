from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
from utils import *


app = FastAPI()

# Generate Token and Insert it into RDB with username, capacity, aggregation method
@app.post("/generate_token/")
async def generate_token(username: str, capa: int, method: str):

    # Generate token
    token = generate_token()

    # Insert token into RDB
    token2rdb(token, username, capa, method)

    return {"token": token}


# upload file and if capacity is full, aggregating model.
@app.post("/upload_file/")
async def upload_file(token: str, username: str, file: UploadFile):
    db = ndb_connect()
    collection = db["posts"]

    if file is None:
        raise HTTPException(status_code=404, detail="File not found")

    model = file.file.read()
    data = {"token" : token, "username" : username, "model" : model}

    collection.insert_one(data)

    # check capacity
    capa = rdb2capa(token)
    cnt = collection.count_documents({"token" : token})
    if cnt == capa:
        # aggregate model
        aggregate_model(token)
        collection.delete_many({"token" : token})
    
    return {"message" : "upload success"}

# refresh request button
@app.post("/refresh_request/")
async def refresh_request(token: str):
    db = ndb_connect()
    collection = db["posts"]
    collection.delete_many({"token" : token})

    return {"message" : "refresh request success"}


@app.get("/download_model/")
async def download_model(username: str):
    models = ndb2userModels(ndb2userTokens(username))
    return models


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
