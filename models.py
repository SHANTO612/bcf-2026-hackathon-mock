from pydantic import BaseModel
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field

# API Request/Response Models
class ParseRequest(BaseModel):
    text: str
    llm: str

class ParseResponse(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    found_in_database: bool
    company: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    database: str

# LangChain Output Schema
class ContactInfo(LCBaseModel):
    name: Optional[str] = Field(description="The person's full name")
    email: Optional[str] = Field(description="The email address")
    phone: Optional[str] = Field(description="The phone number")
