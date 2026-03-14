import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import preprocess_data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "price_paid_records.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    (
        data,
        property_type_encoder,
        old_new_encoder,
        duration_encoder,
        town_city_encoder,
        district_encoder,
        county_encoder,
    ) = preprocess_data(str(data_path))

    X = data[[
        "Property Type",
        "Old/New",
        "Duration",
        "Town/City",
        "District",
        "County",
        "Year"
    ]]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=20,max_depth=10, random_state=42 , n_jobs=-1)),
    ]

    results = []
    best_model = None
    best_r2 = float("-inf")

    for name, model in models:
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)

        if metrics["R2 Score"] > best_r2:
            best_r2 = metrics["R2 Score"]
            best_model = model

    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv(models_dir / "model_metrics.csv", index=False)

    joblib.dump(best_model, models_dir / "house_price_model.pkl")
    joblib.dump(property_type_encoder, models_dir / "property_type_encoder.pkl")
    joblib.dump(old_new_encoder, models_dir / "old_new_encoder.pkl")
    joblib.dump(duration_encoder, models_dir / "duration_encoder.pkl")
    joblib.dump(town_city_encoder, models_dir / "town_city_encoder.pkl")
    joblib.dump(district_encoder, models_dir / "district_encoder.pkl")
    joblib.dump(county_encoder, models_dir / "county_encoder.pkl")


if __name__ == "__main__":
    main()