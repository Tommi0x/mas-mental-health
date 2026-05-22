from pydantic import BaseModel


class Diagnosis(BaseModel):
    agent_name: str
    model_used: str
    disease: str
    explanation: str
