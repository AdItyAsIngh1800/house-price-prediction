from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HouseInput
from database.db import create_table, get_connection

app = FastAPI(title="House Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

print("Starting API...")
print("BASE_DIR:", BASE_DIR)
print("MODELS_DIR:", MODELS_DIR)

create_table()
print("Database table check complete.")

model = joblib.load(MODELS_DIR / "house_price_model.pkl")
property_type_encoder = joblib.load(MODELS_DIR / "property_type_encoder.pkl")
old_new_encoder = joblib.load(MODELS_DIR / "old_new_encoder.pkl")
duration_encoder = joblib.load(MODELS_DIR / "duration_encoder.pkl")
town_city_encoder = joblib.load(MODELS_DIR / "town_city_encoder.pkl")
district_encoder = joblib.load(MODELS_DIR / "district_encoder.pkl")
county_encoder = joblib.load(MODELS_DIR / "county_encoder.pkl")

print("Model and encoders loaded successfully.")

FEATURES = [
    "Property Type",
    "Old/New",
    "Duration",
    "Town/City",
    "District",
    "County",
    "Year",
]


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseInput):
    try:
        input_data = pd.DataFrame({
            "Property Type": [property_type_encoder.transform([data.property_type])[0]],
            "Old/New": [old_new_encoder.transform([data.old_new])[0]],
            "Duration": [duration_encoder.transform([data.duration])[0]],
            "Town/City": [town_city_encoder.transform([data.town_city])[0]],
            "District": [district_encoder.transform([data.district])[0]],
            "County": [county_encoder.transform([data.county])[0]],
            "Year": [data.year],
        })

        input_data = input_data[FEATURES]
        prediction = float(model.predict(input_data)[0])

        # Save prediction to database
        conn = get_connection()
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
                predicted_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.property_type,
                data.old_new,
                data.duration,
                data.town_city,
                data.district,
                data.county,
                data.year,
                prediction,
            ),
        )

        conn.commit()
        conn.close()

        return {
            "predicted_price": prediction,
            "message": "Prediction successful and logged to database",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid category value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predictions")
def get_predictions():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                property_type,
                old_new,
                duration,
                town_city,
                district,
                county,
                year,
                predicted_price,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 20
            """
        )

        rows = cursor.fetchall()
        conn.close()

        predictions = [
            {
                "id": row[0],
                "property_type": row[1],
                "old_new": row[2],
                "duration": row[3],
                "town_city": row[4],
                "district": row[5],
                "county": row[6],
                "year": row[7],
                "predicted_price": row[8],
                "created_at": row[9],
            }
            for row in rows
        ]

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")