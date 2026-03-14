import pandas as pd


def preprocess_data(file_path: str):
    data = pd.read_csv(file_path)

    data["Date of Transfer"] = pd.to_datetime(data["Date of Transfer"])
    data["Year"] = data["Date of Transfer"].dt.year
    data["Month"] = data["Date of Transfer"].dt.month
    data["Quarter"] = data["Date of Transfer"].dt.quarter

    data = data.drop(
        columns=[
            "Transaction unique identifier",
            "PPDCategory Type",
            "Record Status - monthly file only",
        ],
        errors="ignore",
    )

    data = data.dropna()
    data = data[data["Price"] > 1000]

    categorical_columns = [
        "Property Type",
        "Old/New",
        "Duration",
        "Town/City",
        "District",
        "County",
    ]

    for col in categorical_columns:
        data[col] = data[col].astype(str).str.strip()

    data["district_avg_price"] = data.groupby("District")["Price"].transform("mean")
    data["county_avg_price"] = data.groupby("County")["Price"].transform("mean")
    data["town_avg_price"] = data.groupby("Town/City")["Price"].transform("mean")

    return data