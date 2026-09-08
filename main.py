from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pandas as pd
import joblib


# =========================================
# CREATE API
# =========================================

app = FastAPI(
    title="Sahkar Seva Setu AI API",
    description="AI demand forecasting API for Sahkar Seva Setu",
    version="1.0"
)


# =========================================
# ALLOW WEBSITE TO CALL API
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# HUGGING FACE MODEL
# =========================================

MODEL_REPO = "Hetvee2712/sahkar-seva-setu-demand-model"

MODEL_FILE = "sahkar_seva_setu_demand_model.pkl"


# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

# Load trained model
model = joblib.load(model_path)

print("✅ Model loaded successfully!")


# =========================================
# INPUT DATA STRUCTURE
# =========================================

class PredictionInput(BaseModel):

    district: str
    area: str
    service_type: str
    day_of_week: str

    month: int
    season: str
    holiday: int

    active_workers: int

    previous_day_demand: float
    demand_last_7_days: float
    demand_last_30_days: float

    emergency_requests: int

    avg_booking_value: float
    cancelled_requests: int

    completion_rate: float
    avg_response_time: float


# =========================================
# HOME PAGE
# =========================================

@app.get("/")
def home():

    return {
        "message": "Sahkar Seva Setu AI Demand Forecasting API",
        "status": "running"
    }


# =========================================
# PREDICTION ENDPOINT
# =========================================

@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{
        "district": data.district,
        "area": data.area,
        "service_type": data.service_type,
        "day_of_week": data.day_of_week,
        "month": data.month,
        "season": data.season,
        "holiday": data.holiday,
        "active_workers": data.active_workers,
        "previous_day_demand": data.previous_day_demand,
        "demand_last_7_days": data.demand_last_7_days,
        "demand_last_30_days": data.demand_last_30_days,
        "emergency_requests": data.emergency_requests,
        "avg_booking_value": data.avg_booking_value,
        "cancelled_requests": data.cancelled_requests,
        "completion_rate": data.completion_rate,
        "avg_response_time": data.avg_response_time
    }])


    prediction = model.predict(input_data)[0]


    return {
        "predicted_demand": round(float(prediction), 0)
    }
