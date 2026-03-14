from pydantic import BaseModel


class HouseInput(BaseModel):
    property_type: str
    old_new: str
    duration: str
    town_city: str
    district: str
    county: str
    year: int