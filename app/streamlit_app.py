from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# API config
# -----------------------------
API_BASE_URL = "http://127.0.0.1:8000"

# -----------------------------
# Load encoders only for UI labels/options
# -----------------------------
property_type_encoder = joblib.load(MODELS_DIR / "property_type_encoder.pkl")
old_new_encoder = joblib.load(MODELS_DIR / "old_new_encoder.pkl")
duration_encoder = joblib.load(MODELS_DIR / "duration_encoder.pkl")
town_city_encoder = joblib.load(MODELS_DIR / "town_city_encoder.pkl")
district_encoder = joblib.load(MODELS_DIR / "district_encoder.pkl")
county_encoder = joblib.load(MODELS_DIR / "county_encoder.pkl")

# -----------------------------
# Load dataset for dashboard
# -----------------------------
data = pd.read_csv(DATA_DIR / "price_paid_records.csv" , nrows=50000)
data["Date of Transfer"] = pd.to_datetime(data["Date of Transfer"])
data["Year"] = data["Date of Transfer"].dt.year

# -----------------------------
# Load model metrics if available
# -----------------------------
metrics_path = MODELS_DIR / "model_metrics.csv"
comparison_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

# -----------------------------
# Friendly display mappings
# -----------------------------
property_type_display_map = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flat",
}

old_new_display_map = {
    "N": "Old",
    "Y": "New",
}

duration_display_map = {
    "F": "Freehold",
    "L": "Leasehold",
}

# -----------------------------
# Formatter helpers
# -----------------------------
def format_property_type(x: object) -> str:
    return str(property_type_display_map.get(str(x), str(x)))

def format_old_new(x: object) -> str:
    return str(old_new_display_map.get(str(x), str(x)))

def format_duration(x: object) -> str:
    return str(duration_display_map.get(str(x), str(x)))

# -----------------------------
# API helper functions
# -----------------------------
def predict_via_api(payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_prediction_history() -> list[dict]:
    response = requests.get(f"{API_BASE_URL}/predictions", timeout=10)
    response.raise_for_status()
    return response.json().get("predictions", [])

# -----------------------------
# UI
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("Predict house prices using historical property transaction data.")

st.sidebar.header("Project Info")
st.sidebar.write("Built using Python, Pandas, Scikit-learn, Streamlit, and FastAPI.")
st.sidebar.write("Best Model: Random Forest Regressor")
st.sidebar.write("Backend: FastAPI")
st.sidebar.write("Logging: SQLite database")

st.sidebar.markdown("### Features Used")
st.sidebar.write("""
- Property Type
- Old/New
- Duration
- Town/City
- District
- County
- Year
""")

st.info("This app sends user input to a FastAPI backend, which generates predictions and logs them in a database.")

st.markdown("## Project Workflow")
st.write("""
1. Load historical property transaction data  
2. Clean and preprocess dataset  
3. Encode categorical variables  
4. Train regression models  
5. Compare model performance  
6. Use FastAPI backend for prediction  
7. Log predictions in database  
8. Display results in Streamlit  
""")

# -----------------------------
# Dataset preview
# -----------------------------
with st.expander("View Sample Dataset"):
    st.dataframe(data.head(10))

st.markdown("## Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox(
        "Property Type",
        options=property_type_encoder.classes_.tolist(),
        format_func=format_property_type,
    )

    old_new = st.selectbox(
        "Old/New",
        options=old_new_encoder.classes_.tolist(),
        format_func=format_old_new,
    )

    duration = st.selectbox(
        "Duration",
        options=duration_encoder.classes_.tolist(),
        format_func=format_duration,
    )

    year = st.number_input(
        "Year of Transfer",
        min_value=1990,
        max_value=2035,
        value=2015,
        step=1,
    )

with col2:
    town_city = st.selectbox("Town/City", options=town_city_encoder.classes_.tolist())
    district = st.selectbox("District", options=district_encoder.classes_.tolist())
    county = st.selectbox("County", options=county_encoder.classes_.tolist())

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    payload = {
        "property_type": property_type,
        "old_new": old_new,
        "duration": duration,
        "town_city": town_city,
        "district": district,
        "county": county,
        "year": year,
    }

    try:
        result = predict_via_api(payload)
        prediction = result["predicted_price"]

        st.markdown("## Prediction Result")
        st.success(f"Estimated House Price: £{prediction:,.2f}")

        if "message" in result:
            st.caption(result["message"])

        st.markdown("### Selected Property Details")
        user_input_display = pd.DataFrame({
            "Property Type": [property_type_display_map.get(property_type, property_type)],
            "Old/New": [old_new_display_map.get(old_new, old_new)],
            "Duration": [duration_display_map.get(duration, duration)],
            "Town/City": [town_city],
            "District": [district],
            "County": [county],
            "Year": [year],
        })
        st.dataframe(user_input_display)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Start the API with: uvicorn api.main:app --reload")
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("detail", str(e))
        except Exception:
            error_detail = str(e)
        st.error(f"API error: {error_detail}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# -----------------------------
# Recent prediction history
# -----------------------------
st.markdown("## Recent Prediction History")

if st.button("Load Recent Predictions"):
    try:
        history = fetch_prediction_history()
        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df)
        else:
            st.info("No predictions found in the database yet.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Start the API with: uvicorn api.main:app --reload")
    except Exception as e:
        st.error(f"Could not load prediction history: {str(e)}")

# -----------------------------
# Dataset insights
# -----------------------------
st.markdown("## Dataset Insights")

price_by_year = data.groupby("Year")["Price"].mean().reset_index()
st.markdown("### Average House Price by Year")
st.line_chart(price_by_year.set_index("Year"))

property_counts = data["Property Type"].value_counts().rename(index=property_type_display_map)
st.markdown("### Property Type Distribution")
st.bar_chart(property_counts)

# -----------------------------
# Model comparison
# -----------------------------
st.markdown("## Model Comparison")

if not comparison_df.empty:
    st.dataframe(comparison_df)
else:
    st.info("Model comparison metrics not available yet. Run training first.")

# -----------------------------
# Limitations
# -----------------------------
st.markdown("## Limitations")
st.warning("""
This model does not use features such as bedrooms, bathrooms, or property area,
because those fields are not present in the chosen dataset.

Predictions are based mainly on:
- property type
- old/new status
- duration
- location
- year of transfer
""")

# -----------------------------
# Project summary
# -----------------------------
st.markdown("## Project Summary")
st.success("""
This project demonstrates:
- Data preprocessing
- Feature engineering
- Regression modeling
- Model evaluation
- FastAPI backend integration
- Database logging
- User-friendly ML application design
""")