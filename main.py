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
    description="AI demand forecasting and workforce capacity intelligence API",
    version="3.0"
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

    # Location and service
    district: str
    area: str
    service_type: str
    day_of_week: str

    # Time
    month: int
    season: str
    holiday: int

    # Workforce
    active_workers: int

    # IMPORTANT:
    # This will eventually come from the website
    # using worker schedules + existing bookings.
    available_capacity: float

    # Historical demand
    previous_day_demand: float
    demand_last_7_days: float
    demand_last_30_days: float

    # Operational information
    emergency_requests: int
    avg_booking_value: float
    cancelled_requests: int
    completion_rate: float
    avg_response_time: float

# =========================================
# WORKFORCE ALLOCATION INPUT
# =========================================

class NearbyCooperative(BaseModel):
    name: str
    available_capacity: float


class WorkforceAllocationInput(BaseModel):

    predicted_demand: float
    current_capacity: float
    current_workers: int

    nearby_cooperatives: list[NearbyCooperative]

# =========================================
# HOME
# =========================================

@app.get("/")
def home():

    return {
        "message": "Sahkar Seva Setu AI Demand Forecasting API",
        "status": "running",
        "version": "3.0"
    }


# =========================================
# HEALTH CHECK
# =========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =========================================
# DEMAND LEVEL
# =========================================

def get_demand_level(predicted_demand, recent_demand):

    if recent_demand <= 0:
        if predicted_demand > 0:
            return "HIGH"
        return "LOW"

    ratio = predicted_demand / recent_demand

    if ratio >= 1.30:
        return "VERY HIGH"

    elif ratio >= 1.10:
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

    if recent_average <= 0:
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
# CAPACITY STATUS
# =========================================

def get_capacity_status(predicted_demand, available_capacity):

    if available_capacity <= 0:

        if predicted_demand > 0:
            return "SHORTAGE"

        return "NO_ACTIVE_CAPACITY"

    utilisation = (
        predicted_demand / available_capacity
    ) * 100

    if utilisation > 100:
        return "SHORTAGE"

    elif utilisation >= 85:
        return "HIGH PRESSURE"

    elif utilisation >= 60:
        return "HEALTHY"

    else:
        return "EXCESS CAPACITY"


# =========================================
# INSIGHT
# =========================================

def generate_insight(
    predicted_demand,
    available_capacity,
    utilisation,
    status,
    trend,
    service_type,
    area,
    emergency_requests
):

    if status == "SHORTAGE":

        insight = (
            f"Expected demand for {service_type} in {area} "
            f"is higher than the currently available service capacity."
        )

    elif status == "HIGH PRESSURE":

        insight = (
            f"Expected demand for {service_type} in {area} "
            f"is close to the available workforce capacity."
        )

    elif status == "HEALTHY":

        insight = (
            f"Current workforce capacity appears sufficient to "
            f"handle the expected {service_type} demand in {area}."
        )

    else:

        insight = (
            f"Available workforce capacity is comfortably above "
            f"the expected {service_type} demand in {area}."
        )

    # Add trend
    if trend == "RISING":

        insight += " Demand is showing a rising trend."

    elif trend == "FALLING":

        insight += " Demand is showing a falling trend."

    # Add emergency pressure
    if emergency_requests >= 5:

        insight += (
            " Emergency requests are also relatively high."
        )

    return insight


# =========================================
# RECOMMENDATION
# =========================================

def generate_recommendation(
    predicted_demand,
    available_capacity,
    status,
    trend
):

    if status == "SHORTAGE":

        gap = predicted_demand - available_capacity

        return (
            f"Additional service capacity of approximately "
            f"{round(gap, 1)} job-equivalents may be needed. "
            f"Consider reallocating suitable available workers "
            f"from nearby lower-demand areas."
        )

    elif status == "HIGH PRESSURE":

        return (
            "Keep additional workforce capacity on standby "
            "and monitor incoming requests closely."
        )

    elif status == "HEALTHY":

        if trend == "RISING":

            return (
                "Current capacity is sufficient, but demand is rising. "
                "Monitor workforce availability for further increases."
            )

        return (
            "Current workforce capacity appears adequate. "
            "Continue normal workforce allocation."
        )

    else:

        return (
            "Current capacity is more than sufficient. "
            "Potential excess capacity can be utilized in "
            "other higher-demand areas."
        )


# =========================================
# PREDICTION ENDPOINT
# =========================================

@app.post("/predict")
def predict(data: PredictionInput):

    # -----------------------------------------
    # PREPARE DATA FOR XGBOOST MODEL
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
    # ML PREDICTION
    # -----------------------------------------

    prediction = model.predict(input_data)[0]

    predicted_demand = max(
        0,
        round(float(prediction))
    )


    # -----------------------------------------
    # RECENT DEMAND
    # -----------------------------------------

    recent_demand = (
        data.previous_day_demand
        + data.demand_last_7_days
        + data.demand_last_30_days
    ) / 3


    # -----------------------------------------
    # DEMAND LEVEL
    # -----------------------------------------

    demand_level = get_demand_level(
        predicted_demand,
        recent_demand
    )


    # -----------------------------------------
    # TREND
    # -----------------------------------------

    trend = get_trend(
        predicted_demand,
        data.previous_day_demand,
        data.demand_last_7_days,
        data.demand_last_30_days
    )


    # -----------------------------------------
    # CAPACITY UTILISATION
    # -----------------------------------------

    if data.available_capacity > 0:

        utilisation = (
            predicted_demand
            / data.available_capacity
        ) * 100

    else:

        utilisation = 999.0


    # -----------------------------------------
    # CAPACITY STATUS
    # -----------------------------------------

    status = get_capacity_status(
        predicted_demand,
        data.available_capacity
    )


    # -----------------------------------------
    # INSIGHT
    # -----------------------------------------

    insight = generate_insight(
        predicted_demand,
        data.available_capacity,
        utilisation,
        status,
        trend,
        data.service_type,
        data.area,
        data.emergency_requests
    )


    # -----------------------------------------
    # RECOMMENDATION
    # -----------------------------------------

    recommendation = generate_recommendation(
        predicted_demand,
        data.available_capacity,
        status,
        trend
    )


    # -----------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------

    return {

        "predicted_demand": predicted_demand,

        "demand_level": demand_level,

        "trend": trend,

        "active_workers": data.active_workers,

        "available_capacity": round(
            float(data.available_capacity),
            2
        ),

        "utilisation_percent": round(
            float(utilisation),
            1
        ),

        "status": status,

        "insight": insight,

        "recommendation": recommendation
    }
# =========================================
# WORKFORCE ALLOCATION ENDPOINT
# =========================================

@app.post("/workforce-allocation")
def workforce_allocation(data: WorkforceAllocationInput):

    # -----------------------------------------
    # CALCULATE LOCAL CAPACITY GAP
    # -----------------------------------------

    shortage = max(
        0,
        data.predicted_demand - data.current_capacity
    )


    # -----------------------------------------
    # CASE 1: LOCAL CAPACITY IS SUFFICIENT
    # -----------------------------------------

    if shortage == 0:

        return {
            "predicted_demand": data.predicted_demand,
            "current_capacity": data.current_capacity,
            "current_workers": data.current_workers,
            "shortage": 0,
            "status": "SUFFICIENT",
            "allocation_decision": "NO_REALLOCATION",
            "allocations": [],
            "insight": (
                "Current workforce capacity is sufficient "
                "for the expected demand."
            ),
            "recommendation": (
                "Maintain the current workforce allocation."
            )
        }


    # -----------------------------------------
    # CASE 2: SHORTAGE EXISTS
    # -----------------------------------------

    remaining_shortage = shortage

    allocations = []


    # Sort nearby cooperatives by
    # highest available capacity first

    nearby = sorted(
        data.nearby_cooperatives,
        key=lambda x: x.available_capacity,
        reverse=True
    )


    # -----------------------------------------
    # ALLOCATE CAPACITY
    # -----------------------------------------

    for cooperative in nearby:

        if remaining_shortage <= 0:
            break

        transferable_capacity = min(
            cooperative.available_capacity,
            remaining_shortage
        )

        if transferable_capacity > 0:

            allocations.append({
                "cooperative": cooperative.name,
                "capacity_to_reallocate": round(
                    transferable_capacity,
                    2
                )
            })

            remaining_shortage -= transferable_capacity


    # -----------------------------------------
    # CASE 3: ENOUGH NEARBY CAPACITY
    # -----------------------------------------

    if remaining_shortage <= 0:

        allocation_decision = "REALLOCATE"

        status = "SHORTAGE"

        insight = (
            f"Expected demand exceeds local capacity by "
            f"{round(shortage, 2)} capacity units. "
            "Nearby cooperatives have sufficient spare capacity "
            "to support the shortage."
        )

        recommendation = (
            "Reallocate suitable workforce capacity from "
            "nearby cooperatives before adding new workforce."
        )


    # -----------------------------------------
    # CASE 4: NEARBY CAPACITY IS NOT ENOUGH
    # -----------------------------------------

    else:

        allocation_decision = "REALLOCATE_AND_ADD"

        status = "SHORTAGE"

        insight = (
            f"Expected demand exceeds local capacity by "
            f"{round(shortage, 2)} capacity units. "
            f"Nearby cooperatives can cover only part of "
            f"the shortage; approximately "
            f"{round(remaining_shortage, 2)} capacity units "
            "may still be required."
        )

        recommendation = (
            "Use available nearby cooperative capacity first, "
            "then arrange additional workforce capacity."
        )


    # -----------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------

    return {

        "predicted_demand": data.predicted_demand,

        "current_capacity": data.current_capacity,

        "current_workers": data.current_workers,

        "shortage": round(
            shortage,
            2
        ),

        "status": status,

        "allocation_decision": allocation_decision,

        "allocations": allocations,

        "remaining_shortage": round(
            max(0, remaining_shortage),
            2
        ),

        "insight": insight,

        "recommendation": recommendation
    }
