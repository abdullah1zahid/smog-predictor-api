import os
import datetime
import pytz  # NAYA IMPORT - timezone handle karne ke liye
import requests
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lahore Smog Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "aqi_model.pkl" if os.path.exists("aqi_model.pkl") else "lahore_smog_predictor_lgb.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in the directory!")

CSV_PATH = "lahore-us embassy, pakistan-air-quality.csv"
if os.path.exists(CSV_PATH):
    recent_df = pd.read_csv(CSV_PATH)
    recent_df.columns = recent_df.columns.str.strip().str.lower()
    recent_df['date'] = pd.to_datetime(recent_df['date'])
    recent_df = recent_df.sort_values('date').reset_index(drop=True)
    recent_df['pm25'] = pd.to_numeric(recent_df['pm25'], errors='coerce')
    recent_df['pm25'] = recent_df['pm25'].interpolate(method='linear').bfill().ffill()
    recent_values = recent_df['pm25'].tail(14).values
else:
    recent_values = np.array([45.0, 50.0, 42.0, 38.0, 40.0, 35.0, 34.0])

API_TOKEN = "aa135a1e3762b8aea87354ef5e90b19c73b85b14"

# NAYI LINE - Pakistan timezone define kar rahe hain (server ki UTC clock ki bajaye)
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')

feature_cols = [
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 
    'pm25_rolling_7', 'month', 'day_of_year', 'is_smog_season'
]

def get_aqi_category(pm25_value: float):
    if pm25_value <= 50:
        return "Good", "🟢", "Air quality is satisfactory. Enjoy outdoor activities."
    elif pm25_value <= 100:
        return "Moderate", "🟡", "Acceptable air quality. Sensitive individuals should take minor precautions."
    elif pm25_value <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "Children, elderly, and people with respiratory issues should limit outdoor activity."
    elif pm25_value <= 200:
        return "Unhealthy", "🔴", "Everyone should reduce prolonged outdoor exertion. Wear a mask outdoors."
    elif pm25_value <= 300:
        return "Very Unhealthy", "🟣", "Health alert! Avoid outdoor activity. Wear an N95 mask if you must go outside."
    else:
        return "Hazardous", "🟤", "Emergency health warning! Stay strictly indoors."

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Lahore Smog Predictor API is running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/predict")
def predict_tomorrow():
    url = f"https://api.waqi.info/feed/lahore/?token={API_TOKEN}"
    try:
        res = requests.get(url, timeout=10).json()
    except Exception as e:
        return {"error": "Failed to connect to WAQI API", "details": str(e)}

    if res.get("status") != "ok":
        return {"error": "Could not fetch live data", "details": res}

    current_pm25 = res["data"]["iaqi"].get("pm25", {}).get("v", None)
    if current_pm25 is None:
        current_pm25 = res["data"]["aqi"]

    # YEH LINE BADLI HAI - ab Pakistan ke waqt se "aaj" nikal rahe hain, server ki UTC se nahi
    today = datetime.datetime.now(PAKISTAN_TZ)
    tomorrow = today + datetime.timedelta(days=1)

    input_data = {
        'pm25_lag1': float(current_pm25),
        'pm25_lag2': float(recent_values[-1]),
        'pm25_lag3': float(recent_values[-2]),
        'pm25_rolling_7': float(recent_values[-7:].mean()),
        'month': tomorrow.month,
        'day_of_year': tomorrow.timetuple().tm_yday,
        'is_smog_season': int(tomorrow.month in [10, 11, 12, 1, 2])
    }

    df_input = pd.DataFrame([input_data])[feature_cols]
    predicted_pm25 = float(model.predict(df_input)[0])
    category, color, advice = get_aqi_category(predicted_pm25)

    return {
        "city": "Lahore",
        "current_pm25_today": round(float(current_pm25), 1),
        "prediction_date": tomorrow.strftime("%d-%b-%Y"),
        "predicted_pm25_tomorrow": round(predicted_pm25, 1),
        "category": category,
        "indicator": color,
        "advice": advice
    }
