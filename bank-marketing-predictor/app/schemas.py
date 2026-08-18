# app/schemas.py
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    loan: str
    campaign: int = Field(ge=1, le=50)


class PredictionResponse(BaseModel):
    probability: float          # prob de "yes"
    prediction: str             # "yes" / "no"
    classification: str         # texto amigable


class PredictionHistoryResponse(BaseModel):
    id: int
    created_at: str
    age: int
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    loan: str
    campaign: int
    probability: float
    prediction: str
    classification: str