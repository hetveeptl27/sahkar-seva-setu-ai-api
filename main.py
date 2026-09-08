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
    description="AI demand forecasting and workforce insight API for Sahkar Seva Setu",
    version="2.0"
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

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

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
        "status": "running",
        "version": "2.0"
    }


# =========================================
# DEMAND LEVEL
# =========================================

def get_demand_level(predicted_demand, active_workers):

    if active_workers <= 0:
        return "VERY HIGH"

    ratio = predicted_demand / active_workers

    if ratio >= 1.50:
        return "VERY HIGH"

    elif ratio >= 1.20:
        return "HIGH"

    elif ratio >= 0.80:
        return "MODERATE"

    else:
        return "LOW"


# =========================================
# TREND ANALYSIS
# =========================================

def get_trend(predicted_demand, demand_last_7_days, demand_last_30_days):

    recent_average = (
        demand_last_7_days + demand_last_30_days
    ) / 2

    if recent_average == 0:
        return "STABLE"

    change = (
        (predicted_demand - recent_average)
        / recent_average
    ) * 100

    if change >= 10:
        return "RISING"

    elif change <= -10:
        return "FALLING"

    else:
        return "STABLE"


# =========================================
# PREDICTION ENDPOINT
# =========================================

@app.post("/predict")
def predict(data: PredictionInput):

    # -----------------------------------------
    # PREPARE INPUT FOR MODEL
    # -----------------------------------------

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

    # -----------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------

    prediction = model.predict(input_data)[0]

    predicted_demand = max(0, round(float(prediction)))


    # -----------------------------------------
    # DEMAND LEVEL
    # -----------------------------------------

    demand_level = get_demand_level(
        predicted_demand,
        data.active_workers
    )


    # -----------------------------------------
    # TREND
    # -----------------------------------------

    trend = get_trend(
        predicted_demand,
        data.demand_last_7_days,
        data.demand_last_30_days
    )


    # -----------------------------------------
    # WORKFORCE GAP
    # -----------------------------------------

    worker_gap = max(
        0,
        predicted_demand - data.active_workers
    )


    # -----------------------------------------
    # OPERATIONAL INSIGHT
    # -----------------------------------------

    if worker_gap > 0 and trend == "RISING":

        insight = (
            f"High and rising demand expected for "
            f"{data.service_type} in {data.area}. "
            f"Approximately {worker_gap} additional "
            f"workers may be required."
        )

        recommendation = (
            f"Consider reallocating approximately "
            f"{worker_gap} workers from nearby "
            f"lower-demand areas or cooperatives."
        )

    elif worker_gap > 0:

        insight = (
            f"Expected demand may exceed currently "
            f"available workforce by approximately "
            f"{worker_gap} workers."
        )

        recommendation = (
            f"Consider arranging approximately "
            f"{worker_gap} additional workers "
            f"to handle expected demand."
        )

    elif trend == "RISING":

        insight = (
            f"Demand for {data.service_type} in "
            f"{data.area} is showing a rising trend."
        )

        recommendation = (
            "Monitor workforce availability and "
            "prepare additional capacity if demand continues to rise."
        )

    elif trend == "FALLING":

        insight = (
            f"Demand for {data.service_type} in "
            f"{data.area} is showing a falling trend."
        )

        recommendation = (
            "Current workforce capacity appears sufficient; "
            "excess capacity can potentially be utilized elsewhere."
        )

    else:

        insight = (
            f"Demand for {data.service_type} in "
            f"{data.area} is expected to remain relatively stable."
        )

        recommendation = (
            "Maintain current workforce allocation and continue monitoring demand."
        )


    # -----------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------

    return {
        "predicted_demand": predicted_demand,
        "demand_level": demand_level,
        "trend": trend,
        "active_workers": data.active_workers,
        "worker_gap": worker_gap,
        "insight": insight,
        "recommendation": recommendation
    }
