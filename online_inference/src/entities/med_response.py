from pydantic import BaseModel


class MedicalResponse(BaseModel):
    result: int
