@app.get("/predict")
def predict_tomorrow():
    # Open-Meteo Air Quality API (Lahore coordinates: 31.5497, 74.3436)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.5497&longitude=74.3436&current=pm2_5"
    
    try:
        res = requests.get(url, timeout=10).json()
        current_pm25 = res.get("current", {}).get("pm2_5")
        if current_pm25 is None:
            return {"error": "Could not fetch current PM2.5", "details": res}
    except Exception as e:
        return {"error": "Failed to connect to Air Quality API", "details": str(e)}

    # Pakistan time
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
