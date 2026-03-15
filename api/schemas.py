from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    property_type: Literal["D", "S", "T", "F"]
    old_new: Literal["Y", "N"]
    duration: Literal["F", "L"]

    town_city: str = Field(..., min_length=2, max_length=100)
    district: str = Field(..., min_length=2, max_length=100)
    county: str = Field(..., min_length=2, max_length=100)

    year: int = Field(..., ge=1995, le=2035)

    @field_validator("town_city", "district", "county")
    @classmethod
    def clean_strings(cls, value: str) -> str:
        return value.strip().upper()


class PredictionResponse(BaseModel):
    predicted_price: float