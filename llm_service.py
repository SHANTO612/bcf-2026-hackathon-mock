from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import GEMINI_API_KEY, OPENAI_API_KEY
from models import ContactInfo

class APIKeyMissingError(Exception):
    """Exception raised when an API key is missing."""
    pass

def extract_with_langchain(text: str, model_name: str, provider: str) -> dict:
    """Extract contact info using LangChain with either Gemini or OpenAI."""
    
    llm = None
    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise APIKeyMissingError("Gemini API Key is missing")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise APIKeyMissingError("OpenAI API Key is missing")
        llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

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
        # Re-raise APIKeyMissingError, handle others
        if isinstance(e, APIKeyMissingError):
            raise e
        print(f"LangChain error ({provider}): {e}")
        # Return empty/null structure if extraction fails but key was present
        return {"name": None, "email": None, "phone": None}
