import os
import numpy as np
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env
load_dotenv()


def get_embedding_function():
    ''' Get the embedding function for ChromaDB using Google Generative AI.'''
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GEMINI_API_KEY") )
