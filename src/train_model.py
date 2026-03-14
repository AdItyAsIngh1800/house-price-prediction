from pathlib import Path
from typing import cast

import joblib
import mlflow
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import preprocess_data


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    metrics = {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "CV Mean R2": cv_scores.mean()
    }

    return metrics, pipeline


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "price_paid_records.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri("file:" + str((base_dir / "mlruns").resolve()))
    mlflow.set_experiment("House Price Prediction")

    # Autolog sklearn params / metrics / model
    mlflow_sklearn.autolog(log_input_examples=False, log_model_signatures=True)

    data = preprocess_data(str(data_path))

    if len(data) > 50000:
        data = data.sample(n=50000, random_state=42)

    feature_columns = [
        "Property Type",
        "Old/New",
        "Duration",
        "Town/City",
        "District",
        "County",
        "Year",
        "Month",
        "Quarter",
        "district_avg_price",
        "county_avg_price",
        "town_avg_price",
    ]

    X = data[feature_columns]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = [
        "Property Type",
        "Old/New",
        "Duration",
        "Town/City",
        "District",
        "County",
    ]

    numeric_features = [
        "Year",
        "Month",
        "Quarter",
        "district_avg_price",
        "county_avg_price",
        "town_avg_price",
    ]
    X[numeric_features] = X[numeric_features].astype("float64")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )),
    ]

    results = []
    best_pipeline = None
    best_r2 = float("-inf")

    for name, model in models:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        with mlflow.start_run(run_name=name):
            metrics, trained_pipeline = evaluate_model(
                name, pipeline, X_train, X_test, y_train, y_test
            )

            mlflow.set_tag("model_family", name)
            mlflow.log_metric("custom_mae", metrics["MAE"])
            mlflow.log_metric("custom_mse", metrics["MSE"])
            mlflow.log_metric("custom_rmse", metrics["RMSE"])
            mlflow.log_metric("custom_r2", metrics["R2 Score"])
            mlflow.log_metric("custom_cv_mean_r2", metrics["CV Mean R2"])
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            results.append(metrics)

            if metrics["R2 Score"] > best_r2:
                best_r2 = metrics["R2 Score"]
                best_pipeline = trained_pipeline

    print("\nRunning Random Forest hyperparameter tuning...")

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    rf_param_grid = {
        "model__n_estimators": [50, 100, 150, 200],
        "model__max_depth": [6, 8, 10, 12, None],
        "model__min_samples_split": [5, 10, 20],
        "model__min_samples_leaf": [2, 5, 10],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_grid,
        n_iter=10,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    with mlflow.start_run(run_name="Random Forest Tuned"):
        rf_search.fit(X_train, y_train)

        best_rf = cast(Pipeline, rf_search.best_estimator_)
        rf_predictions = best_rf.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_predictions)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_rmse = rf_mse ** 0.5
        rf_r2 = r2_score(y_test, rf_predictions)

        tuned_metrics = {
            "Model": "Random Forest Tuned",
            "MAE": rf_mae,
            "MSE": rf_mse,
            "RMSE": rf_rmse,
            "R2 Score": rf_r2,
            "CV Mean R2": rf_search.best_score_
        }

        mlflow.set_tag("model_family", "Random Forest Tuned")
        mlflow.log_metric("custom_mae", tuned_metrics["MAE"])
        mlflow.log_metric("custom_mse", tuned_metrics["MSE"])
        mlflow.log_metric("custom_rmse", tuned_metrics["RMSE"])
        mlflow.log_metric("custom_r2", tuned_metrics["R2 Score"])
        mlflow.log_metric("custom_cv_mean_r2", tuned_metrics["CV Mean R2"])

        # Log best tuned params explicitly
        for k, v in rf_search.best_params_.items():
            mlflow.log_param(k, v)

        results.append(tuned_metrics)

        if rf_r2 > best_r2:
            best_r2 = rf_r2
            best_pipeline = best_rf

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)

    results_df.to_csv(models_dir / "model_metrics.csv", index=False)
    joblib.dump(best_pipeline, models_dir / "house_price_pipeline.pkl")

    print("\nBest pipeline saved as: models/house_price_pipeline.pkl")
    print("MLflow tracking saved under: mlruns/")


if __name__ == "__main__":
    main()