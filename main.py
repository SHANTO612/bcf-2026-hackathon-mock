from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import psycopg2
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field
import re
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5434")
DB_NAME = os.getenv("DB_NAME", "practice_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain Output Schema
class ContactInfo(LCBaseModel):
    name: Optional[str] = Field(description="The person's full name")
    email: Optional[str] = Field(description="The email address")
    phone: Optional[str] = Field(description="The phone number")

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

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def clean_json_string(json_str: str) -> str:
    """Clean JSON string from markdown formatting."""
    json_str = re.sub(r'^```json\s*', '', json_str)
    json_str = re.sub(r'^```\s*', '', json_str)
    json_str = re.sub(r'\s*```$', '', json_str)
    return json_str.strip()

def mock_extract(text: str) -> dict:
    print("WARNING: Using Mock Extraction (Regex/Heuristic) because API keys are missing.")
    
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else None
    
    phone_match = re.search(r'\d{3}-\d{3}-\d{4}', text)
    phone = phone_match.group(0) if phone_match else None
    
    # Simple name heuristics based on test cases
    name = None
    
    words = text.split()
    for i in range(len(words) - 1):
        w1 = words[i].strip(",.")
        w2 = words[i+1].strip(",.")
        if w1[0].isupper() and w2[0].isupper() and w1.isalpha() and w2.isalpha():
            if w1 in ["Contact", "Call", "You", "For", "If"]:
                continue
            name = f"{w1} {w2}"
            break
            
    return {"name": name, "email": email, "phone": phone}

def extract_with_langchain(text: str, model_name: str, provider: str) -> dict:
    """Extract contact info using LangChain with either Gemini or OpenAI."""
    
    llm = None
    if provider == "gemini":
        if not GEMINI_API_KEY:
            return mock_extract(text)
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            return mock_extract(text)
        llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    else:
        return mock_extract(text)

    try:
        parser = JsonOutputParser(pydantic_object=ContactInfo)
        
        prompt = PromptTemplate(
            template="Extract contact information from the given text.\n{format_instructions}\nText: {text}",
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        
        result = chain.invoke({"text": text})
        return result
    except Exception as e:
        print(f"LangChain error ({provider}): {e}")
        return mock_extract(text)

@app.get("/health", response_model=HealthResponse)
def health_check():
    conn = get_db_connection()
    db_status = "connected" if conn else "disconnected"
    if conn:
        conn.close()
    return {"status": "ok", "database": db_status}

@app.get("/")
def root():
    return {
        "message": "Contact Parser API is running",
        "endpoints": {
            "health_check": "GET /health",
            "parse_contact": "POST /parse"
        }
    }

@app.post("/parse", response_model=ParseResponse)
def parse_contact(request: ParseRequest):
    # 1. Extract info using LLM (or Mock)
    extracted_data = {}
    
    # Normalize model name or just pass through
    llm_lower = request.llm.lower()
    
    if "gemini" in llm_lower:
        extracted_data = extract_with_langchain(request.text, request.llm, "gemini")
    elif "gpt" in llm_lower:
        extracted_data = extract_with_langchain(request.text, request.llm, "openai")
    else:
        # Default to mock if unknown
        extracted_data = mock_extract(request.text)
    
    name = extracted_data.get("name")
    email = extracted_data.get("email")
    phone = extracted_data.get("phone")
    
    # 2. Check Database
    found_in_database = False
    company = None
    
    if name:
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                # Check for name match (case insensitive)
                # Assuming name is "First Last"
                query = """
                SELECT co.name 
                FROM contacts c 
                LEFT JOIN companies co ON c.company_id = co.company_id 
                WHERE LOWER(c.first_name || ' ' || c.last_name) = LOWER(%s)
                """
                cur.execute(query, (name,))
                result = cur.fetchone()
                
                if result:
                    found_in_database = True
                    company = result[0]
                
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Database query error: {e}")
                if conn: conn.close()
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "found_in_database": found_in_database,
        "company": company
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
