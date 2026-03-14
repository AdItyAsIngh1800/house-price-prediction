from pathlib import Path
from datetime import datetime
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
DB_PATH = BASE_DIR / "house_predictions.db"

# -----------------------------
# Load ML pipeline
# -----------------------------
pipeline = joblib.load(MODEL_PATH)

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

    input_df = pd.DataFrame([{
        "Property Type": request.property_type,
        "Old/New": request.old_new,
        "Duration": request.duration,
        "Town/City": request.town_city,
        "District": request.district,
        "County": request.county,
        "Year": request.year
    }])

    prediction = pipeline.predict(input_df)[0]

    predicted_price = float(round(prediction, 2))

    # log prediction
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


# -----------------------------
# Prediction History
# -----------------------------
@app.get("/predictions")
def get_predictions():

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
    conn.close()

    return {"predictions": df.to_dict(orient="records")}