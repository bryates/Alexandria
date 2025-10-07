''' This module crates the database connection to the ChromaDB instance and uses Google Generative AI to embed the text.'''
import os
import shutil
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google.generativeai import configure
import google.generativeai as genai
from get_embedding_function import get_embedding_function

# Load .env
load_dotenv()
configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chromadb/')
DATA_PATH = os.getenv("DATA_PATH", "data/")


def load_documents():
    ''' Load data from the DATA_PATH directory.'''
    # loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf")
    # loader = DirectoryLoader('', glob="README*.md")
    loader = DirectoryLoader('', glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    ''' Split the documents into chunks.'''
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    return texts


def calculate_chunk_ids(chunks):
    ''' Calculate unique IDs for each chunk.'''
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Assuming chunk has metadata with 'page_id'
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page', 0)
        current_page_id = f"{source}:{page}"

        if current_page_id != last_page_id:
            current_chunk_index = 0
        else:
            current_chunk_index += 1
        last_page_id = current_page_id

        chunk.metadata['id'] = f'{current_page_id}:{current_chunk_index}'
    
    return chunks


def save_to_chroma(chunks, batch_size=10):
    """Save documents to ChromaDB using precomputed Gemini embeddings for speed."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()

    collection = chroma_client.get_or_create_collection(
        name="results"
        # no embedding function needed here because we'll provide vectors
    )

    existing_docs = collection.get(include=["metadatas"])
    existing_ids = set(m["id"] for m in existing_docs["metadatas"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if not new_chunks:
        print("No new documents to add.")
        return

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i+batch_size]
        docs = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]
        ids = [c.metadata["id"] for c in batch]

        # precompute embeddings for the whole batch
        embeddings = embedding_fn.embed_documents(docs)  # __call__ returns list of lists

        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings  # pass precomputed vectors
        )

        print(f"Added batch {i // batch_size + 1}: {len(batch)} chunks")

    print(f"Finished adding {len(new_chunks)} chunks to ChromaDB at {CHROMA_PATH}.")



def clean_chunks(chunks):
    ''' Clean the text chunks by removing images and non-printable characters.'''
    import re
    cleaned = []
    for c in chunks:
        text = c.page_content if hasattr(c, "page_content") else str(c)
        # Remove HTML img tags
        text = re.sub(r'<img[^>]*>', '', text)
        # Remove Markdown images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # Remove non-printable/control characters
        text = ''.join(ch for ch in text if ch.isprintable())
        # Skip empty chunks
        if text.strip():
            cleaned.append(text)
    return cleaned

docs = load_documents()
chunks = split_text(docs)
print([c.metadata['id'] for c in calculate_chunk_ids(chunks)])
# exit()
# chunks = clean_chunks(chunks)
print(f'Loaded {len(docs)} documents and split into {len(chunks)} chunks.')
save_to_chroma(chunks)
