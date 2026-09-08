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
    description="AI demand forecasting and workforce planning API",
    version="2.0"
)


# =========================================
# CORS
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
# INPUT DATA
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
# HOME
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
# TREND
# =========================================

def get_trend(
    predicted_demand,
    previous_day_demand,
    demand_last_7_days,
    demand_last_30_days
):

    recent_average = (
        previous_day_demand
        + demand_last_7_days
        + demand_last_30_days
    ) / 3

    if recent_average == 0:
        return "STABLE"

    percentage_change = (
        (predicted_demand - recent_average)
        / recent_average
    ) * 100

    if percentage_change >= 10:
        return "RISING"

    elif percentage_change <= -10:
        return "FALLING"

    else:
        return "STABLE"


# =========================================
# INSIGHT GENERATOR
# =========================================

def generate_insight(
    predicted_demand,
    active_workers,
    emergency_requests,
    completion_rate,
    trend,
    demand_level,
    service_type,
    area
):

    capacity_ratio = (
        predicted_demand / active_workers
        if active_workers > 0
        else float("inf")
    )

    if demand_level == "VERY HIGH":

        insight = (
            f"Very high demand is expected for {service_type} "
            f"in {area}. Forecasted demand is substantially "
            f"above currently available workforce capacity."
        )

    elif demand_level == "HIGH":

        insight = (
            f"High demand is expected for {service_type} "
            f"in {area}. Workforce capacity may come under pressure."
        )

    elif demand_level == "MODERATE":

        insight = (
            f"Moderate demand is expected for {service_type} "
            f"in {area}. Current capacity appears relatively balanced."
        )

    else:

        insight = (
            f"Low demand is expected for {service_type} "
            f"in {area}. Available capacity may be sufficient."
        )

    # Add trend information
    if trend == "RISING":

        insight += " Demand is also showing a rising trend."

    elif trend == "FALLING":

        insight += " Demand is showing a falling trend."

    # Add emergency pressure
    if emergency_requests >= 5:

        insight += (
            " Emergency requests are relatively high, "
            "indicating additional service pressure."
        )

    # Add completion pressure
    if completion_rate < 80:

        insight += (
            " The current completion rate is low, "
            "so capacity planning should be monitored closely."
        )

    return insight


# =========================================
# RECOMMENDATION GENERATOR
# =========================================

def generate_recommendation(
    predicted_demand,
    active_workers,
    demand_level,
    trend
):

    capacity_gap = max(
        0,
        predicted_demand - active_workers
    )

    if demand_level == "VERY HIGH":

        if capacity_gap > 0:

            return (
                f"Prioritize additional workforce capacity. "
                f"Consider reallocating suitable workers from "
                f"nearby lower-demand areas or cooperatives."
            )

        return (
            "Maintain high workforce readiness and monitor incoming requests closely."
        )

    elif demand_level == "HIGH":

        if trend == "RISING":

            return (
                "Prepare additional workforce capacity and "
                "monitor demand closely for further increases."
            )

        return (
            "Keep additional workforce capacity on standby "
            "to handle demand fluctuations."
        )

    elif demand_level == "MODERATE":

        return (
            "Current workforce allocation appears adequate. "
            "Continue monitoring demand and worker availability."
        )

    else:

        return (
            "Current capacity appears sufficient. "
            "Excess capacity may be utilized in other higher-demand areas."
        )


# =========================================
# PREDICTION
# =========================================

@app.post("/predict")
def predict(data: PredictionInput):

    # -----------------------------------------
    # PREPARE MODEL INPUT
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

    predicted_demand = max(
        0,
        round(float(prediction))
    )


    # -----------------------------------------
    # ANALYSIS
    # -----------------------------------------

    demand_level = get_demand_level(
        predicted_demand,
        data.active_workers
    )


    trend = get_trend(
        predicted_demand,
        data.previous_day_demand,
        data.demand_last_7_days,
        data.demand_last_30_days
    )


    capacity_gap = max(
        0,
        predicted_demand - data.active_workers
    )


    insight = generate_insight(
        predicted_demand,
        data.active_workers,
        data.emergency_requests,
        data.completion_rate,
        trend,
        demand_level,
        data.service_type,
        data.area
    )


    recommendation = generate_recommendation(
        predicted_demand,
        data.active_workers,
        demand_level,
        trend
    )


    # =========================================
    # FINAL RESPONSE
    # =========================================

    return {

        "predicted_demand": predicted_demand,

        "demand_level": demand_level,

        "trend": trend,

        "active_workers": data.active_workers,

        "capacity_gap": capacity_gap,

        "insight": insight,

        "recommendation": recommendation
    }
