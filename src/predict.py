import joblib
import pandas as pd
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    model = joblib.load(models_dir / "house_price_model.pkl")

    sample_input = pd.DataFrame({
        "Property Type": [0],
        "Old/New": [0],
        "Duration": [0],
        "Town/City": [10],
        "District": [10],
        "County": [5],
        "Year": [2015]
    })

    prediction = model.predict(sample_input)[0]
    print(f"Predicted Price: £{prediction:,.2f}")


if __name__ == "__main__":
    main()