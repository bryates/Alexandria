"""
RAG module for querying an existing ChromaDB collection
and summarizing results with Google Gemini.
"""
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from update_db import add_to_chroma, split_text
from alexandria.get_embedding_function import get_embedding_function

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chromadb/')

PATHS = [
    'https://raw.githubusercontent.com/bryates/Alexandria/refs/heads/main/README.md',
    'https://raw.githubusercontent.com/bryates/Agent-Lander/refs/heads/main/README.md',
    'https://raw.githubusercontent.com/bryates/climateboost/refs/heads/main/README.md',
    'https://raw.githubusercontent.com/bryates/ttbarML/refs/heads/master/README.md'
]

embedding_fn = get_embedding_function()

# Load existing ChromaDB collection
chromadb = Chroma(
    collection_name="results",
    embedding_function=embedding_fn,
    persist_directory=CHROMA_PATH
)
collection_name = "results"
collection = chromadb._client.get_collection(collection_name)

for path in PATHS:
    docs = WebBaseLoader(path).load()
    for doc in docs:
        doc.metadata['source'] = path.split('/')[4]  # repo name
    chunks = split_text(docs)
    add_to_chroma(chunks)  


def query_chroma(chroma_client, query, k=5):
    """Perform a similarity search in ChromaDB."""
    results = chroma_client.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Document: {doc.page_content[:300]}...\n")
        print(f"Metadata: {doc.metadata}\nDistance: {score}")
        print("-"*50)
    return results


# Load existing ChromaDB collection
# chromadb = Chroma(
#     collection_name="results",
#     embedding_function=embedding_fn,
#     persist_directory=CHROMA_PATH
# )


# collection = None
# for c in collections:
#     if c.name == 'results':
#         collection = c
#         break


if collection:
    print("Documents in 'results':", collection.count())
    # Optionally print first 3 document IDs
    docs = collection.get(include=[])['ids']
    print("First 20 document IDs:", docs[:20])
else:
    print("Collection 'results' not found in this path.")

try:
    print(f"Loaded collection '{collection_name}' with {collection.count()} documents")
except Exception as e:
    raise RuntimeError(f"Failed to load collection '{collection_name}': {e}")

query_text = "Find all projects where I used GPT, LLMs, NLP, or generative AI, and summarize each in 2-3 bullets."
query_text = '''This job requires experience with machine learning, natural language processing (NLP), and large language models (LLMs) such as GPT.
Also Deep Learning, Python, and cloud platforms like AWS or GCP.'''
query_vec = embedding_fn.embed_query(query_text)
if not isinstance(query_vec, np.ndarray):
    query_vec = np.array(query_vec)

results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

context_text = "\n".join([f"- {doc} (source: {meta.get('source', 'unknown')})"
                        for doc, meta in zip(results["documents"][0], results["metadatas"][0])])

PROMPT_TEMPLATE = """You are an expert at summarizing project documentation.
Given the following context:

{context}

# Answer the user's question concisely:
Based on this job posting:
"{question}"
find the tope 2 projects that best align and summarize each in 2-3 bullets.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
response = llm.invoke([HumanMessage(content=prompt)])
print("\n=== LLM Summary ===")
print(response.content)
