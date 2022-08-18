from typing import List
from fastapi import Query
from pydantic import BaseModel

class Text(BaseModel):
    text: str = Query(None, min_length=1)

class PredictPayload(BaseModel):
    texts: List[Text]


    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "questions mount mtn nigeria alleged payment n618 7bn taxes firs mtn nigeria communications plc claim paid sum n618 7 billion direct indirect taxes."},
                    {"text": "Uefa Opens Proceedings against Barcelona, Juventus and Real Madrid Over European Super League Plan. Uefa has opened disciplinary proceedings against Barcelona, Juventus and Real Madrid over their involvement in the proposed European Super League."},
                ]
            }
        }