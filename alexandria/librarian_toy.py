'''Example code to call Gemini API using the Google Generative AI SDK in Python.'''
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Please set GEMINI_API_KEY in .env")

# Configure the SDK
genai.configure(api_key=api_key)

# Choose a model (free tier-compatible one, e.g. “gemini-2.5-flash” or similar)
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Make a request
response = model.generate_content("Explain in simple terms how AI works.")

print(response.text)


# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or "gemini-2.5-pro", etc.
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

# Simple message call
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Write me a haiku about LangChain and Gemini.")
]

response = llm.invoke(messages)
print(response.content)
