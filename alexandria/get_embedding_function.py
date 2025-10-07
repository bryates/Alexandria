import os
import numpy as np
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env
load_dotenv()


# def get_embedding_function():
#     ''' Get the embedding function for ChromaDB using Google Generative AI.'''
#     # return embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ.get('GEMINI_API_KEY'))
#     return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GEMINI_API_KEY"])

def get_embedding_function():
    ''' Get the embedding function for ChromaDB using Google Generative AI.'''
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GEMINI_API_KEY") )

'''
class GeminiEmbedding:
    def __init__(self, api_key):
        self.fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Chroma expects this signature.
        'input' is a list of strings (documents or queries),
        return a list of embedding vectors (list of floats).
        """
        out = []
        for text in input:
            vec = self.fn.embed_query(text)  # or embed_documents if you prefer
            # flatten numpy arrays to list
            if isinstance(vec, np.ndarray):
                out.append(vec.tolist())
            elif isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], np.ndarray):
                out.append(vec[0].tolist())
            else:
                out.append(list(vec))
        return out

    # def embed_query(self, text):
    #     vec = self.fn.embed_query(text)
    #     # vec may be np.array or [np.array], flatten it to 1D list
    #     if isinstance(vec, np.ndarray):
    #         print("Query embedding type:", type(vec), "length:", len(vec))
    #         return vec.tolist()
    #     if isinstance(vec, list):
    #         # flatten one level if it's a list of np.array
    #         vec0 = vec[0]
    #         if isinstance(vec0, np.ndarray):
    #             print("Query embedding type:", type(vec), "length:", len(vec))
    #             return vec0.tolist()
    #         elif isinstance(vec0, list):
    #             return vec0  # already a list of floats
    #     # fallback
    #     print("Query embedding type:", type(vec), "length:", len(vec))
    #     return list(vec)

    # def embed_documents(self, texts):
    #     vecs = self.fn.embed_documents(texts)
    #     out = []
    #     for v in vecs:
    #         if isinstance(v, np.ndarray):
    #             out.append(v.tolist())
    #         elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], np.ndarray):
    #             out.append(v[0].tolist())
    #         else:
    #             out.append(list(v))
    #     return out
    
    def name(self):
        return "gemini_embedding"
'''


# def get_embedding_function():
#     ''' Get the embedding function for ChromaDB using Google Generative AI.'''
#     return GeminiEmbedding(os.environ.get("GEMINI_API_KEY"))