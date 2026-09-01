# 🌫️ Lahore Smog Predictor — Next-Day PM2.5 Forecast & Health Advisory

A machine learning system that predicts tomorrow's PM2.5 air pollution level for Lahore, Pakistan, and translates that prediction into a plain-language health advisory — built on nearly 5 years of real, publicly available air quality data.

🔗 **Live Demo:** https://smog-vision-lahore.lovable.app

---

## 📌 Problem Statement

Lahore regularly ranks among the most polluted cities in the world, especially during the October–February smog season. Residents currently have no easy way to know in advance whether tomorrow's air will be safe to go outside in. This project turns raw historical pollution data into a forward-looking, actionable forecast.

---

## 📊 Data Source

- **Source:** [World Air Quality Index (WAQI) Historical Data Platform](https://aqicn.org/historical) — an open, research-grade air quality monitoring network covering 12,000+ stations worldwide
- **Station:** Lahore US Consulate monitoring station
- **Range:** ~1,845 days of daily PM2.5 readings
- **Live data:** Real-time current-day PM2.5 pulled via the free WAQI REST API (`api.waqi.info`)

This is genuine, continuously recorded environmental monitoring data — not a synthetic or scraped dataset.

---

## 🧠 Approach

### Feature Engineering
Since this is a time-series problem, the model relies on:
- **Lag features**: PM2.5 from 1, 2, and 3 days ago
- **7-day rolling average**: smooths short-term noise, captures the underlying trend
- **Calendar features**: month, day-of-year
- **Domain knowledge flag**: `is_smog_season` (Oct–Feb), since Punjab's crop-burning and temperature-inversion season behaves very differently from the rest of the year

### Rigorous Evaluation (not just "high accuracy")
- **Chronological train/test split** (not random) — to avoid data leakage, since shuffling time-series data would let the model "see the future" during training
- **Baseline comparison**: a naive "tomorrow = today" persistence model was built first. The ML model was only considered useful because it outperformed this baseline — a standard practice in applied ML that guards against reporting misleading accuracy
- **Three models compared**: Linear Regression, Random Forest, XGBoost
- **SHAP (Explainable AI)** used to confirm which features actually drive the prediction (yesterday's PM2.5 and the 7-day rolling average dominate, as expected physically)

### From Prediction to Action
The raw PM2.5 number is converted into a standard AQI health category (Good → Hazardous) with a specific, actionable health recommendation for each level — e.g. "wear an N95 mask outdoors" or "stay indoors."

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Data source | WAQI Historical Data Platform + live REST API |
| Data processing | pandas, numpy |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Environment | Google Colab (Python) |

---

## 📁 Project Structure

```
lahore-smog-predictor/
│
├── aqi_model_training.ipynb   # Full notebook: cleaning, features, training, evaluation, SHAP
├── live_prediction.py          # Pulls live PM2.5 via API and predicts tomorrow's value
├── aqi_model.pkl                # Trained Random Forest model
└── README.md
```

---

## ▶️ Running the Live Prediction

```bash
pip install requests pandas scikit-learn joblib

python live_prediction.py
```

The script fetches Lahore's current live PM2.5 reading via the WAQI API, combines it with recent historical trend data, and outputs tomorrow's predicted PM2.5 along with a health category and advisory.

**Note:** You'll need your own free API token from [aqicn.org/data-platform/register](https://aqicn.org/data-platform/register) — registration only requires a valid email.

---

## 🔮 Future Improvements

- Extend forecast horizon to 3–7 days instead of just tomorrow
- Incorporate wind speed/direction and temperature inversion data for better smog-season accuracy
- Deploy as a live web dashboard with daily auto-refreshing predictions
- Expand to other Punjab cities (Multan, Faisalabad, Rahim Yar Khan) for regional comparison

---

## 📬 Contact

Built by Abdullah Zahid — feel free to connect on [LinkedIn]([(https://www.linkedin.com/in/abdullah-zahid89/)] or reach out with feedback!
