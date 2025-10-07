"""Module to add documents to an existing ChromaDB instance."""
import os
from langchain_community.document_loaders.directory import DirectoryLoader
import chromadb
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function

# Load environment variables
load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chromadb/')
DATA_PATH = os.getenv('DATA_PATH', 'data/')

# -------------------------------
# Document loading and splitting
# -------------------------------
def load_documents():
    """Load data from DATA_PATH directory."""
    loader = DirectoryLoader(DATA_PATH, glob='*.md')
    documents = loader.load()
    return documents

def split_text(documents):
    """Split the documents into chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    """Assign unique IDs to each chunk based on source and page."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', 0)
        current_page_id = f"{source}:{page}"

        if current_page_id != last_page_id:
            current_chunk_index = 0
        else:
            current_chunk_index += 1
        last_page_id = current_page_id

        chunk.metadata['id'] = f"{current_page_id}:{current_chunk_index}"
    
    return chunks


# -------------------------------
# Add chunks to existing ChromaDB
# -------------------------------
def add_to_chroma(chunks):
    """Add chunks to existing ChromaDB, skipping duplicates."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection('results')  # Use existing collection only
    embedding_fn = get_embedding_function()

    # Get existing IDs
    try:
        existing_ids = set(collection.get(include=["ids"])["ids"])
    except Exception:
        existing_ids = set()

    # Assign unique IDs
    chunks = calculate_chunk_ids(chunks)

    # Filter out duplicates
    new_chunks = [c for c in chunks if c.metadata['id'] not in existing_ids]

    if not new_chunks:
        print("No new documents to add.")
        return
    print([c.metadata['id'] for c in new_chunks])
    collection.add(
        documents=[c.page_content for c in new_chunks],
        metadatas=[c.metadata for c in new_chunks],
        ids=[c.metadata['id'] for c in new_chunks],
        embeddings=embedding_fn.embed_documents([c.page_content for c in new_chunks])  # Precompute embeddings
    )

    print(f"Added {len(new_chunks)} new documents to ChromaDB at '{CHROMA_PATH}'.")

# -------------------------------
# Optional cleaning
# -------------------------------
def clean_chunks(chunks):
    """Remove images and non-printable characters from text chunks."""
    import re
    cleaned = []
    for c in chunks:
        text = getattr(c, 'page_content', str(c))
        text = re.sub(r'<img[^>]*>', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = ''.join(ch for ch in text if ch.isprintable())
        if text.strip():
            cleaned.append(text)
    return cleaned

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_text(docs)
    calculate_chunk_ids(chunks)
    print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks.")
    add_to_chroma(chunks)
