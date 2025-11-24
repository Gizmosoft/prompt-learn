from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

# Method 1: Using langchain-google-genai (recommended with your current packages)
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize a model to test the connection
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key
    )
    print("Successfully connected to Gemini via LangChain!")
    print(f"Model: {llm.model_name}")
except Exception as e:
    print(f"Error: {e}")

# Method 2: List available models using direct API call with httpx (already in your requirements)
print("\nAvailable Gemini models:")
try:
    response = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key}
    )
    if response.status_code == 200:
        models = response.json().get("models", [])
        for model in models:
            print(f"- {model.get('name')}")
    else:
        print(f"Error fetching models: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")