from fastapi import FastAPI, Response, File, UploadFile
from fastapi.responses import RedirectResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import ApplicationException
from sensor.logger import logging
from sensor.pipeline import train
from sensor.pipeline.train import TrainPipeline
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import read_yaml_file, load_object
import os
import pandas as pd
from uvicorn import run as app_run
import io

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_route(file: UploadFile):
    try:
        #get data from user csv file
        contents = await file.read()
        #conver csv file to dataframe
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        
        #return the dataframe as JSON response
        return df.to_json(orient='records')
        
    except Exception as e:
        raise Response(f"Error Occurred! {e}")

def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']

def main():
    try:
        #set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__=="__main__":
    #main()
    # set_env_variable(env_file_path)
    app_run(app, port=8000)
