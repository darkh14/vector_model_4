from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LoadingDataResponse(BaseModel):
    loading_id: str
    message: str


# class TokenRequest(BaseModel):
#     email: EmailStr

# class TokenCheck(BaseModel):
#     token: str

# class TokenData(BaseModel):
#     email: EmailStr
#     token: str
#     expiration: datetime
#     type: str
#     created_at: datetime
#     last_used: Optional[datetime]
    