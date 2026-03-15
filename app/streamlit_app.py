from pathlib import Path

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
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

API_BASE_URL = "http://127.0.0.1:8000"

# -----------------------------
# Load dataset for dashboard
# -----------------------------
data = pd.read_csv(DATA_DIR / "price_paid_records.csv", nrows=20000)
data["Date of Transfer"] = pd.to_datetime(data["Date of Transfer"])
data["Year"] = data["Date of Transfer"].dt.year

# -----------------------------
# Load model metrics if available
# -----------------------------
metrics_path = MODELS_DIR / "model_metrics.csv"
comparison_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

# -----------------------------
# Load feature importance if available
# -----------------------------
importance_path = MODELS_DIR / "feature_importance.csv"
importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()

# -----------------------------
# Friendly display mappings
# -----------------------------
property_type_display = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flat"
}

old_new_display = {
    "N": "Old",
    "Y": "New"
}

duration_display = {
    "F": "Freehold",
    "L": "Leasehold"
}

# -----------------------------
# UI
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("Predict house prices using historical property transaction data.")

st.sidebar.header("Project Info")
st.sidebar.write("Built using Python, Pandas, Scikit-learn, FastAPI, and Streamlit.")
st.sidebar.write("Model used: Linear Regression")

st.sidebar.markdown("### Features Used")
st.sidebar.write("""
- Property Type
- Old/New
- Duration
- Town/City
- District
- County
- Year
- Location-based engineered features
""")

st.info(
    "This system predicts house prices using a Linear Regression model served through FastAPI."
)

st.markdown("## Project Workflow")
st.write("""
1. Load historical property transaction data  
2. Clean and preprocess dataset  
3. Train regression models  
4. Compare model performance  
5. Select best model for deployment  
6. Use FastAPI backend for prediction  
7. Log predictions into database  
8. Display results in Streamlit  
""")

# -----------------------------
# Dataset preview
# -----------------------------
with st.expander("View Sample Dataset"):
    st.dataframe(data.head(10), use_container_width=True)

# -----------------------------
# User Input
# -----------------------------
st.markdown("## Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox(
        "Property Type",
        options=list(property_type_display.keys()),
        format_func=lambda x: property_type_display[x]
    )

    old_new = st.selectbox(
        "Old/New",
        options=list(old_new_display.keys()),
        format_func=lambda x: old_new_display[x]
    )

    duration = st.selectbox(
        "Duration",
        options=list(duration_display.keys()),
        format_func=lambda x: duration_display[x]
    )

    year = st.number_input(
        "Year of Transfer",
        min_value=1995,
        max_value=2035,
        value=2015,
        step=1
    )

with col2:
    town_city = st.selectbox(
        "Town/City",
        options=sorted(data["Town/City"].dropna().astype(str).unique())
    )

    district = st.selectbox(
        "District",
        options=sorted(data["District"].dropna().astype(str).unique())
    )

    county = st.selectbox(
        "County",
        options=sorted(data["County"].dropna().astype(str).unique())
    )

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
        "year": int(year),
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        prediction = result["predicted_price"]

        st.markdown("## Prediction Result")
        st.success(f"Estimated House Price: £{prediction:,.2f}")

        st.markdown("### Selected Property Details")
        display_df = pd.DataFrame({
            "Property Type": [property_type_display[property_type]],
            "Old/New": [old_new_display[old_new]],
            "Duration": [duration_display[duration]],
            "Town/City": [town_city],
            "District": [district],
            "County": [county],
            "Year": [year],
        })
        st.dataframe(display_df, use_container_width=True)

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
# Prediction History
# -----------------------------
st.markdown("## Prediction History")

history_df = pd.DataFrame()

if st.button("Load Recent Predictions"):
    try:
        response = requests.get(f"{API_BASE_URL}/predictions", timeout=10)
        response.raise_for_status()
        history = response.json().get("predictions", [])

        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No predictions found in the database yet.")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Start the API with: uvicorn api.main:app --reload")
    except Exception as e:
        st.error(f"Could not load prediction history: {str(e)}")

# -----------------------------
# Model Monitoring
# -----------------------------
st.markdown("## 📊 Model Monitoring Metrics")

try:
    response = requests.get(f"{API_BASE_URL}/monitoring", timeout=10)
    response.raise_for_status()
    metrics = response.json()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", metrics["total_predictions"])
    col2.metric("Average Price", f"£{metrics['average_price']:,.0f}")
    col3.metric("Highest Price", f"£{metrics['max_price']:,.0f}")
    col4.metric("Lowest Price", f"£{metrics['min_price']:,.0f}")

except Exception as e:
    st.error(f"Could not load monitoring metrics: {str(e)}")

# -----------------------------
# Prediction Distribution
# -----------------------------
st.markdown("### Prediction Price Distribution")

try:
    response = requests.get(f"{API_BASE_URL}/predictions", timeout=10)
    response.raise_for_status()
    history = response.json().get("predictions", [])

    if history:
        history_df = pd.DataFrame(history)

        if "predicted_price" in history_df.columns:
            distribution_df = history_df[["predicted_price"]].copy()
            distribution_df.index = pd.Index(range(1, len(distribution_df) + 1))
            st.bar_chart(distribution_df)
        else:
            st.info("Prediction data is available, but no predicted_price column was found.")
    else:
        st.info("No prediction records available yet for distribution chart.")

except Exception as e:
    st.error(f"Could not load prediction distribution: {str(e)}")

# -----------------------------
# Dataset insights
# -----------------------------
st.markdown("## Dataset Insights")

price_by_year = data.groupby("Year")["Price"].mean().reset_index()
st.markdown("### Average Price by Year")
st.line_chart(price_by_year.set_index("Year"))

property_counts = data["Property Type"].value_counts().rename(index=property_type_display)
st.markdown("### Property Type Distribution")
st.bar_chart(property_counts)

# -----------------------------
# Location-Based Price Insights
# -----------------------------
st.markdown("## 📍 Location-Based Price Insights")

town_price_df = (
    data.groupby("Town/City")["Price"]
    .mean()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

county_price_df = (
    data.groupby("County")["Price"]
    .mean()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

district_price_df = (
    data.groupby("District")["Price"]
    .mean()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 Most Expensive Towns/Cities")
    st.dataframe(town_price_df.head(10), use_container_width=True)

with col2:
    st.markdown("### Top 10 Most Affordable Towns/Cities")
    st.dataframe(
        town_price_df.tail(10).sort_values(by="Price", ascending=True),
        use_container_width=True
    )

st.markdown("### Average Price by County")
st.bar_chart(county_price_df.head(15).set_index("County"))

st.markdown("### Average Price by District")
st.bar_chart(district_price_df.head(15).set_index("District"))

# -----------------------------
# County Filter Analysis
# -----------------------------
st.markdown("### County-Level Detailed Analysis")

selected_county = st.selectbox(
    "Select County for Detailed Analysis",
    options=sorted(data["County"].dropna().astype(str).unique())
)

county_filtered = data[data["County"] == selected_county]

towns_in_county = (
    county_filtered.groupby("Town/City")["Price"]
    .mean()
    .reset_index()
    .sort_values(by="Price", ascending=False)
)

st.dataframe(towns_in_county.head(20), use_container_width=True)

# -----------------------------
# Model comparison
# -----------------------------
st.markdown("## Model Comparison")

if not comparison_df.empty:
    st.dataframe(comparison_df, use_container_width=True)
else:
    st.info("Model comparison metrics not available yet. Run training first.")

# -----------------------------
# Feature Importance
# -----------------------------
st.markdown("## Model Feature Importance")

if not importance_df.empty:
    top_features = importance_df.head(20)
    st.bar_chart(top_features.set_index("feature")["importance"])
else:
    st.info("Feature importance available only when a tree-based model is deployed and feature importance has been exported.")

# -----------------------------
# Limitations
# -----------------------------
st.markdown("## Limitations")
st.warning("""
This model does not include several important real estate features such as bedrooms,
property area, bathrooms, or building condition.

Therefore predictions are based mainly on:
- property type
- ownership duration
- location
- transaction year
""")

# -----------------------------
# Project Summary
# -----------------------------
st.markdown("## Project Summary")
st.success("""
This project demonstrates:

- Data preprocessing
- Feature engineering
- Regression model comparison
- FastAPI model serving
- Streamlit dashboard development
- SQLite database logging
- Docker containerization
- Location-based analytics
- Basic model interpretability
- Model monitoring metrics
""")