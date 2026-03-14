import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(file_path: str):
    data = pd.read_csv(file_path)

    data["Date of Transfer"] = pd.to_datetime(data["Date of Transfer"])
    data["Year"] = data["Date of Transfer"].dt.year

    data = data.drop(columns=[
        "Transaction unique identifier",
        "PPDCategory Type",
        "Record Status - monthly file only"
    ])

    data = data.dropna()
    data = data[data["Price"] > 1000]

    categorical_columns = [
        "Property Type",
        "Old/New",
        "Duration",
        "Town/City",
        "District",
        "County"
    ]

    for col in categorical_columns:
        data[col] = data[col].astype(str).str.strip()

    property_type_encoder = LabelEncoder()
    old_new_encoder = LabelEncoder()
    duration_encoder = LabelEncoder()
    town_city_encoder = LabelEncoder()
    district_encoder = LabelEncoder()
    county_encoder = LabelEncoder()

    data["Property Type"] = property_type_encoder.fit_transform(data["Property Type"])
    data["Old/New"] = old_new_encoder.fit_transform(data["Old/New"])
    data["Duration"] = duration_encoder.fit_transform(data["Duration"])
    data["Town/City"] = town_city_encoder.fit_transform(data["Town/City"])
    data["District"] = district_encoder.fit_transform(data["District"])
    data["County"] = county_encoder.fit_transform(data["County"])

    return (
        data,
        property_type_encoder,
        old_new_encoder,
        duration_encoder,
        town_city_encoder,
        district_encoder,
        county_encoder,
    )