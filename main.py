from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sensor.exception import ApplicationException
from sensor.logger import logging
from sensor.pipeline.train import TrainPipeline
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, SCHEMA_FILE_PATH
from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import read_yaml_file, load_object
import os
import pandas as pd
from uvicorn import run as app_run
import io
import sys
import numpy as np
from flask import Flask, jsonify
from sensor.ml.model.estimator import TargetValueMapping


from sensor.pipeline.S3_upload import S3_upload

app = FastAPI()
templates = Jinja2Templates(directory="templates")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the prediction directory path
PREDICTION_DIR = "./prediction_directory"

@app.get("/")
async def get_index_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_route(file: UploadFile):
    try:
        # Read contents of uploaded CSV file
        contents = await file.read()

        # Convert CSV file to DataFrame
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Create the prediction directory if it doesn't exist
        os.makedirs(PREDICTION_DIR, exist_ok=True)

        # Drop specified columns if schema is available
        schema = read_yaml_file(SCHEMA_FILE_PATH)
        drop_columns = schema.get("drop_columns", [])
        print(drop_columns)

        df.drop(columns=drop_columns, axis=1, inplace=True)
        df.replace({"na": np.nan}, inplace=True)
        X_train = df.drop('class', axis=1)
        df.to_csv(os.path.join(PREDICTION_DIR, "before_preprocessor.csv"), index=False)

        # Apply data transformations if available
        preprocessor = load_object(file_path='preprocessor/preprocessor.pkl')

        # Apply transformations using preprocessor
        X_array = preprocessor.transform(X_train)

        # Load and use the best model for prediction
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return JSONResponse(content={"message": "Model is not available"}, status_code=404)

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        # Make predictions
        y_pred = model.predict(X_array)
        # Map 0 to neg and 1 to pos
        y_pred_mapped = ["neg" if pred == 0 else "pos" for pred in y_pred]

        # Create a DataFrame from y_train_pred
        df1 = pd.DataFrame(y_pred_mapped, columns=['class'])

        # Save the predicted DataFrame to a CSV file
        predicted_csv_path = os.path.join(PREDICTION_DIR, "y_prediction.csv")
        df1.to_csv(predicted_csv_path, index=False)
        
        

        message = "prediction_done"

        return message

    except Exception as e:
        return JSONResponse(content={"message": f"Error occurred! {e}"}, status_code=500)

def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']

@app.get('/train')
def train_pipeline():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        response = {
            'message': 'Training pipeline completed successfully!'
        }
        
        return jsonify(response)
    except ApplicationException as e:
        return str(e)

training_pipeline = TrainPipeline()
training_pipeline.run_pipeline()
    
@app.get('/upload')
def bucket_upload():
    try:
        upload_class=S3_upload()
        upload_class.upload
        
        return "Upload Successful"
    except Exception as e:
        raise ApplicationException(e,sys) 

if __name__ == "__main__":
    # set_env_variable(env_file_path='env.yaml')
    app_run(app, port=8000)