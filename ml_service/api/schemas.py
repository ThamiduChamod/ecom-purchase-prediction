from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    customerId: str = Field(..., description="CustomerID as string")

class PredictResponse(BaseModel):
    customerId: str
    prediction: int
    probability: float
    modelName: str