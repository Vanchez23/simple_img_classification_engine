from urllib import request
from fastapi import FastAPI, Response
from os import environ
import databases

database = databases.Database(r"sqlite:///db/pythonsqlite.db")


app = FastAPI()

@app.on_event("startup")
async def startup():
    # когда приложение запускается устанавливаем соединение с БД
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    # когда приложение останавливается разрываем соединение с БД
    await database.disconnect()

@app.get("/")
async def read_root():
    query = """SELECT * FROM predictions LIMIT 10"""
    return await database.fetch_all(query)

@app.get("/api/predict/{img_link}")
async def predict(img_link: str):
    """ Создает нового пользователя в БД """
    # query = f"""INSERT OR IGNORE INTO predictions (image_link, status) values("{img_link}", "processing")"""
    # prediction_id = await database.execute(query)

    label = await make_prediction(img_link)
    
    query = f"""INSERT OR IGNORE INTO predictions (image_link, image_label, status) values("{img_link}", "{label}", "done")"""
    prediction_id = await database.execute(query)
    
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
    count_deleted_records = await database.execute(query)
    return Response(status_code=204)
    
@app.post("/api/train")
async def train():

    return {"success": True}

async def make_prediction(img_path):
    return "empty"