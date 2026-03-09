from pydantic import BaseModel

class InferResponse(BaseModel):
    img: str
    prob_illegal: float
