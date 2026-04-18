from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
import joblib
import time
import uuid
import numpy as np
import os

MODEL_PATH = os.getenv("MODEL_PATH", "taxi_model.pkl")
ml_model =None
start_time = None
#Model Loading:
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, start_time
    ml_model =joblib.load(MODEL_PATH)
    start_time =time.time()
    print("model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(title="taxi tip Predictor",lifespan=lifespan)#reduced to one as was conflicting with the first app declaration from the suppmental docs provided
#Single Prediction Endpoint (POST /predict)
class taxiInput(BaseModel):
    VendorID: int =Field(..., ge=1, le=2)
    passenger_count: int =Field(..., ge=1, le=6)
    trip_distance: float =Field(..., gt=0)
    RatecodeID: int =Field(..., ge=1, le=6)
    PULocationID: int =Field(..., ge=1)
    DOLocationID: int =Field(..., ge=1)
    payment_type: int =Field(..., ge=1, le=5)
    fare_amount: float =Field(..., ge=0)
    extra: float =Field(..., ge=0)
    mta_tax: float =Field(..., ge=0)
    tolls_amount: float =Field(..., ge=0)
    improvement_surcharge: float =Field(..., ge=0)
    congestion_surcharge: float =Field(..., ge=0)
    Airport_fee: float =Field(..., ge=0)
    trip_duration_minutes: int =Field(..., gt=0)
    trip_speed_mph: float =Field(..., ge=0)
    pickup_hour: int =Field(..., ge=0, le=23)
    pickup_day_of_week: int =Field(..., ge=0, le=6)
    log_trip_distance: float
    fare_per_mile: float
    fare_per_minute: float
    tpep_pickup_datetime_hour: int =Field(..., ge=0, le=23)
    tpep_pickup_datetime_day: int =Field(..., ge=1, le=31)
    tpep_pickup_datetime_month: int =Field(..., ge=1, le=12)
    tpep_dropoff_datetime_hour: int =Field(..., ge=0, le=23)
    tpep_dropoff_datetime_day: int =Field(..., ge=1, le=31)
    tpep_dropoff_datetime_month: int =Field(..., ge=1, le=12)
    store_and_fwd_flag_Y: bool
    pickup_Borough_Brooklyn: bool
    pickup_Borough_EWR: bool
    pickup_Borough_Manhattan: bool
    pickup_Borough_N_A: bool
    pickup_Borough_Queens: bool
    pickup_Borough_Staten_Island: bool
    pickup_Borough_Unknown: bool
    dropoff_Borough_Brooklyn: bool
    dropoff_Borough_EWR: bool
    dropoff_Borough_Manhattan: bool
    dropoff_Borough_N_A: bool
    dropoff_Borough_Queens: bool
    dropoff_Borough_Staten_Island: bool
    dropoff_Borough_Unknown: bool


class PredictionResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str

def to_feature_list(input_data: taxiInput):#healper, since input was so large, will be better if PCA was used to reduced and have less inputs
    return [[
        input_data.VendorID,
        input_data.passenger_count,
        input_data.trip_distance,
        input_data.RatecodeID,
        input_data.PULocationID,
        input_data.DOLocationID,
        input_data.payment_type,
        input_data.fare_amount,
        input_data.extra,
        input_data.mta_tax,
        input_data.tolls_amount,
        input_data.improvement_surcharge,
        input_data.congestion_surcharge,
        input_data.Airport_fee,
        input_data.trip_duration_minutes,
        input_data.trip_speed_mph,
        input_data.pickup_hour,
        input_data.pickup_day_of_week,
        input_data.log_trip_distance,
        input_data.fare_per_mile,
        input_data.fare_per_minute,
        input_data.tpep_pickup_datetime_hour,
        input_data.tpep_pickup_datetime_day,
        input_data.tpep_pickup_datetime_month,
        input_data.tpep_dropoff_datetime_hour,
        input_data.tpep_dropoff_datetime_day,
        input_data.tpep_dropoff_datetime_month,
        input_data.store_and_fwd_flag_Y,
        input_data.pickup_Borough_Brooklyn,
        input_data.pickup_Borough_EWR,
        input_data.pickup_Borough_Manhattan,
        input_data.pickup_Borough_N_A,
        input_data.pickup_Borough_Queens,
        input_data.pickup_Borough_Staten_Island,
        input_data.pickup_Borough_Unknown,
        input_data.dropoff_Borough_Brooklyn,
        input_data.dropoff_Borough_EWR,
        input_data.dropoff_Borough_Manhattan,
        input_data.dropoff_Borough_N_A,
        input_data.dropoff_Borough_Queens,
        input_data.dropoff_Borough_Staten_Island,
        input_data.dropoff_Borough_Unknown
    ]]

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: taxiInput):
    features = to_feature_list(input_data)
    prediction = ml_model.predict(features)[0]
    return PredictionResponse(
    prediction=round(float(prediction), 2),
    prediction_id=str(uuid.uuid4()),
    model_version="1.0.0",
    )

#Batch Prediction
class BatchInput(BaseModel):
    records: List[taxiInput] =Field(..., max_length=100)  

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    start =time.time() 
    predictions = []
    for record in batch.records:
       
        pred = ml_model.predict(to_feature_list(record))[0]
        predictions.append(PredictionResponse(
        prediction=round(float(pred), 2),
        prediction_id=str(uuid.uuid4()),
        model_version="1.0.0",
        ))
    elapsed = (time.time() -start) * 1000
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2),
        )

@app.get("/health")
def health_check():
    print("DEBUG:", ml_model)
    uptime = None
    if start_time is not None:
        uptime = round(time.time() - start_time, 1)
    return {
    "status": "healthy",
    "model_loaded": ml_model is not None,
    "model_version": "1.0.0",
    }


@app.get("/model/info")
def model_info():
    return {
    "model_name": "taxi tip predictor",
    "version": "1.0.0",
    "features":list(taxiInput.model_fields.keys()),
    "metrics": {"MAE ": 1.18, "RMSE": 2.28,"R2":0.64},
    "trained_date": "2026-04-13",
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
    status_code=500,
    content={
    "error": "Internal server error",
    "detail": "An unexpected error occurred. Please try again.",
    },
    )