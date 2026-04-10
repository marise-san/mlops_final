from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class BankMarketingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Numeric fields with simple validations
    age: int = Field(..., ge=18, le=120, description="Age of the client")
    balance: float = Field(..., description="Average yearly balance in euros")
    day_of_week: int = Field(..., ge=1, le=31, alias="day", description="Last contact day of the month")
    duration: float = Field(..., ge=0.0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts performed during this campaign")
    pdays: int = Field(..., ge=-1, description="Number of days since last contact from previous campaign (-1 means not contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign")

    # Categorical fields strictly validated against possible categories
    job: Literal[
        "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", 
        "student", "blue-collar", "self-employed", "retired", "technician", "services"
    ]
    marital: Literal["married", "divorced", "single", "unknown"]
    education: Literal[
        "unknown", "secondary", "primary", "tertiary", "basic.4y", 
        "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree"
    ]
    default: Literal["yes", "no", "unknown"]
    housing: Literal["yes", "no", "unknown"]
    loan: Literal["yes", "no", "unknown"]
    contact: Literal["unknown", "cellular", "telephone"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcome: Literal["unknown", "other", "failure", "success", "nonexistent"]


class PredictionOutput(BaseModel):
    """Schema for returning prediction results"""
    prediction: Literal["yes", "no"]
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence/probability of the prediction class 'yes'")
