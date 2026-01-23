import requests
import json

def test_gemini():
    url = "http://localhost:8000/parse"
    payload = {
        "text": "Please contact Alice Wonderland at alice@example.com",
        "llm": "gemini-2.5-flash"
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini()
