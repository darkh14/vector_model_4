from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class TokenRequest(BaseModel):
    email: EmailStr

class TokenCheck(BaseModel):
    token: str

class TokenData(BaseModel):
    token: str
    expiration: datetime
    type: str
    created_at: datetime
    last_used: Optional[datetime]
    