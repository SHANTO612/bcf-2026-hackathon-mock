from fastapi import FastAPI, HTTPException
from models import ParseRequest, ParseResponse, HealthResponse
from database import check_db_connection, search_contact_by_name
from llm_service import extract_with_langchain, APIKeyMissingError

app = FastAPI()

@app.get("/health", response_model=HealthResponse)
def health_check():
    db_status = check_db_connection()
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
    # 1. Extract info using LLM
    extracted_data = {}
    
    # Normalize model name or just pass through
    llm_lower = request.llm.lower()
    
    try:
        if "gemini" in llm_lower:
            extracted_data = extract_with_langchain(request.text, request.llm, "gemini")
        elif "gpt" in llm_lower:
            extracted_data = extract_with_langchain(request.text, request.llm, "openai")
        else:
             # Default behavior: try Gemini if model name is ambiguous, or raise error
             # For now, let's assume if it's not gpt, it might be gemini or just fail
             # Let's default to Gemini if unknown for this specific hackathon context, 
             # or better, raise an error.
             # Given the user instruction "remove mock regex", we should not fallback to it.
             # Let's try to infer from model name or just fail.
             raise HTTPException(status_code=400, detail="Unsupported model type. Please use a Gemini or GPT model name.")
    except APIKeyMissingError as e:
        # User requested to "response with that also" if AI missing
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Extraction failed: {str(e)}")

    
    name = extracted_data.get("name")
    email = extracted_data.get("email")
    phone = extracted_data.get("phone")
    
    # 2. Check Database
    found_in_database, company = search_contact_by_name(name)
    
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
