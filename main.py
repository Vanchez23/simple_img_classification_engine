from typing import List
import urllib
import time
import os.path as osp
import yaml
import json
from datetime import datetime

from fastapi import FastAPI, Response, Request
import databases
import uvicorn
import torch
import numpy as np
import cv2

from model import Pipeline
from model.dataset import CustomDataset

database = databases.Database(r"sqlite:///db/pythonsqlite.db")

app = FastAPI()
train_process = False
train_id = None
pipeline = None

@app.on_event("startup")
async def startup():
    global train_id, pipeline

    await database.connect()
    with open('model/model_config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    query = f"""SELECT id, model_path FROM trains ORDER BY created_at DESC LIMIT 1"""
    row = await database.fetch_all(query)
    if row:
        train_id = row[0][0]
        if not cfg['checkpoint_cfg']['load_path']:
            cfg['checkpoint_cfg']['load_path'] = row[0][1]
        
    pipeline = Pipeline(cfg)

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
async def read_root():
    query = """SELECT * FROM predictions LIMIT 10"""
    return await database.fetch_all(query)

@app.get("/api/predict")
async def predict(request: Request):
    body = await request.json()
    img_link = body['img_link']
    if train_id is None:
        return {"prediction": None, "exception": "No available models"}

    # try:
    start_prediction = time.time()
    label = pipeline.classifier.predict(load_img(img_link))
    duration = time.time() - start_prediction
    # except Exception as ex:
    #     return {"prediction": None, "exception": ex}

    query = f"""INSERT OR IGNORE INTO predictions (image_link, image_label, prediction_time, status, train_id) \
                    values("{img_link}", "{label}",{duration}, "done", {train_id})"""
    await database.execute(query)
    
    return {"prediction": label}

@app.get("/api/get_past_predictions")
async def get_all_predictions():
    query = f"""SELECT prediction_time, image_link, image_label
                FROM predictions
                WHERE status=="done" """
    predictions = await database.fetch_all(query)
    return predictions

@app.delete("/api/clear_past_predictions", status_code=204)
async def clear_db():
    query = f"""DELETE FROM predictions
                WHERE status=="done" """
    await database.execute(query)
    return Response(status_code=204)
    
@app.post("/api/train")
async def train():
    global train_process, train_id, pipeline
    result = {"success": True, "in_progress": train_process}
    if train_process:
        return {"success": True, "in_progress": train_process}
    train_process = True
    # try:

    metrics = pipeline.run()
    experiment = pipeline._cfg['checkpoint_cfg']['experiment_name']
    model_path = osp.join(pipeline.classifier.save_checkpoint_path, 'best.pth')
    loss = metrics[0]['test/epoch_loss']
    accuracy = metrics[0]['test/epoch_accuracy']
    data_version = 0
    created_at = datetime.strptime(pipeline.classifier.created_at,"%Y_%m_%d_%H_%M_%S").strftime("%Y:%m:%d_%H:%M:%S")
    query = f"""INSERT OR IGNORE INTO trains (experiment, model_path, loss, accuracy, data_version, created_at) 
                values("{experiment}", "{model_path}", {loss}, {accuracy}, "{data_version}", "{created_at}")"""
    train_id = await database.execute(query)
    train_process = False
    # except Exception as ex:
    #     train_process = False
    #     result= {"success": False, "in_progress": train_process, "exception": ex}
    # finally:
    return result

def load_img(img_link: str) -> np.ndarray:
    
    req = urllib.request.urlopen(img_link)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    return img

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000, reload=True)