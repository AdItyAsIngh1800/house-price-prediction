from datetime import datetime
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
DB_PATH = BASE_DIR / "house_predictions.db"
DATA_PATH = BASE_DIR / "data" / "price_paid_records.csv"

# -----------------------------
# Load ML pipeline
# -----------------------------
pipeline = joblib.load(MODEL_PATH)

# -----------------------------
# Load dataset for feature engineering lookups
# -----------------------------
reference_data = pd.read_csv(DATA_PATH, nrows=50000)
reference_data["Date of Transfer"] = pd.to_datetime(reference_data["Date of Transfer"])
reference_data["Year"] = reference_data["Date of Transfer"].dt.year
reference_data["Month"] = reference_data["Date of Transfer"].dt.month
reference_data["Quarter"] = reference_data["Date of Transfer"].dt.quarter

reference_data = reference_data.drop(
    columns=[
        "Transaction unique identifier",
        "PPDCategory Type",
        "Record Status - monthly file only",
    ],
    errors="ignore",
)

reference_data = reference_data.dropna()
reference_data = reference_data[reference_data["Price"] > 1000]

for col in ["Property Type", "Old/New", "Duration", "Town/City", "District", "County"]:
    reference_data[col] = reference_data[col].astype(str).str.strip()

district_price_map = reference_data.groupby("District")["Price"].mean().to_dict()
county_price_map = reference_data.groupby("County")["Price"].mean().to_dict()
town_price_map = reference_data.groupby("Town/City")["Price"].mean().to_dict()

global_avg_price = float(reference_data["Price"].mean())

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="House Price Prediction API")

# -----------------------------
# Request Schema
# -----------------------------
class PredictionRequest(BaseModel):
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    year: int

# -----------------------------
# Initialize SQLite DB
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_type TEXT,
            old_new TEXT,
            duration TEXT,
            town_city TEXT,
            district TEXT,
            county TEXT,
            year INTEGER,
            predicted_price REAL,
            timestamp TEXT
        )
        """
    )

    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def root():
    return {"message": "House Price Prediction API running"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_price(request: PredictionRequest):
    try:
        month = 1
        quarter = 1

        district_avg_price = district_price_map.get(request.district, global_avg_price)
        county_avg_price = county_price_map.get(request.county, global_avg_price)
        town_avg_price = town_price_map.get(request.town_city, global_avg_price)

        input_df = pd.DataFrame([{
            "Property Type": request.property_type,
            "Old/New": request.old_new,
            "Duration": request.duration,
            "Town/City": request.town_city,
            "District": request.district,
            "County": request.county,
            "Year": request.year,
            "Month": month,
            "Quarter": quarter,
            "district_avg_price": district_avg_price,
            "county_avg_price": county_avg_price,
            "town_avg_price": town_avg_price,
        }])

        prediction = pipeline.predict(input_df)[0]
        predicted_price = float(round(prediction, 2))

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                property_type,
                old_new,
                duration,
                town_city,
                district,
                county,
                year,
                predicted_price,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.property_type,
                request.old_new,
                request.duration,
                request.town_city,
                request.district,
                request.county,
                request.year,
                predicted_price,
                datetime.utcnow().isoformat()
            )
        )

        conn.commit()
        conn.close()

        return {"predicted_price": predicted_price}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Prediction History
# -----------------------------
@app.get("/predictions")
def get_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
    conn.close()

    return {"predictions": df.to_dict(orient="records")}