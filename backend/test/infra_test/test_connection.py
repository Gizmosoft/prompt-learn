import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
 

 # Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Send a test message with Langfuse callback
message = [HumanMessage(content="Write a short 10-word poem about technology.")]
response = llm.invoke(message, config={"callbacks": [langfuse_handler]})

print("Gemini Response:\n", response.content)
print("\n✅ Gemini call successful!")
print("✅ LangFuse trace created automatically via callback!")