import sys
import time
import os.path as osp
import yaml
from datetime import datetime

from fastapi import FastAPI, Response, Request
import databases
import uvicorn
from loguru import logger

from pipeline import Pipeline

database = databases.Database(r"sqlite:///db/pythonsqlite.db")

app = FastAPI()

train_process = False
train_id = None
pipeline = None

@app.on_event("startup")
async def startup():
    global train_id, pipeline

    logger.add("main.log", rotation="500 MB")

    await database.connect()
    logger.info("The database is connected")

    with open('pipeline_config.yaml') as f:
        cfg = yaml.safe_load(f)

    # load model    
    if cfg['load_predict_model']:
        # read last train
        query = f"""SELECT id, model_path FROM trains ORDER BY created_at DESC LIMIT 1"""
        row = await database.fetch_all(query)
        if row:
            if cfg['checkpoint_cfg']['load_path']:
                logger.info(f'The model is loading from load_path = {cfg["checkpoint_cfg"]["load_path"]}')
            else:
                cfg['checkpoint_cfg']['load_path'] = row[0][1]
                train_id = row[0][0]
                logger.info(f'The model is loading from train_id == {train_id}')
        else:
            logger.info('Trains are not found')
    else:
        logger.info("A prediction model is not loaded")

    pipeline = Pipeline(cfg)
    logger.info("The pipeline is configured")
    logger.info("Success")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    logger.info("The database is disconnected")

@app.get("/api/predict")
async def predict(request: Request):
    body = await request.json()
    img_link = body['img_link']
    if train_id is None:
        logger.error("No available models")
        return Response("No available models", status_code=500)

    try:
        start_prediction = time.time()
        label = pipeline.predict(img_link)
        duration = time.time() - start_prediction
    except Exception as ex:
        return Response("Internal server error", status_code=500)
    created_at = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    query = f"""INSERT OR IGNORE INTO predictions (image_link, image_label, prediction_time, created_at, train_id) \
                    values("{img_link}", "{label}",{duration}, "{created_at}", {train_id})"""

    await database.execute(query)
    
    logger.info(f"Prediction for {img_link} is {label}")
    return {"prediction": label}

@app.get("/api/get_past_predictions")
async def get_all_predictions():
    query = f"""SELECT prediction_time, image_link, image_label
                FROM predictions"""
    predictions = await database.fetch_all(query)
    return predictions

@app.delete("/api/clear_past_predictions", status_code=204)
async def clear_db():
    query = f"""DELETE FROM predictions"""
    await database.execute(query)
    return Response(status_code=204)
    
@app.post("/api/train")
async def train():
    global train_process, train_id, pipeline
    result = {"success": True, "in_progress": train_process}
    if train_process:
        logger.info("The train in progress")
        return result
    train_process = True
    try:
        logger.info("A train pipeline just started")
        start_train = datetime.now()

        metrics = pipeline.run()
        
        train_duration = str(datetime.now() - start_train)
        experiment = pipeline._cfg['checkpoint_cfg']['experiment_name']
        model_path = osp.join(pipeline.classifier.save_checkpoint_path, 'best.pth')
        loss = metrics[0]['test/epoch/loss']
        accuracy = metrics[0]['test/epoch/accuracy']
        f1_score = metrics[0]['test/epoch/f1_score']
        precision = metrics[0]['test/epoch/precision']
        recall = metrics[0]['test/epoch/recall']
        data_version = 0
        created_at = datetime.strptime(pipeline.classifier.created_at,"%Y_%m_%d_%H_%M_%S").strftime("%Y:%m:%d %H:%M:%S")
        
        query = f"""INSERT OR IGNORE INTO trains (experiment, model_path, loss, accuracy, f1_score, precision, recall, data_version, duration, created_at) 
                    values("{experiment}", "{model_path}", {loss}, {accuracy},{f1_score},{precision},{recall}, "{data_version}","{train_duration}", "{created_at}")"""
        
        train_id = await database.execute(query)
        logger.info(f"The train pipeline is over. Metrics: {metrics}")
        train_process = False
    except Exception:
        train_process = False
        return Response("Internal server error", status_code=500)
    finally:
        return result

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000)