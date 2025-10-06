''' This module crates the database connection to the ChromaDB instance and uses Google Generative AI to embed the text.'''
import os
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google.generativeai import configure
import google.generativeai as genai


# Load .env
load_dotenv()
configure(api_key=os.getenv('GOOGLE_API_KEY'))

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chromadb/')
DATA_PATH = os.getenv('DATA_PATH', 'data/')


def load_documents():
    ''' Load data from the DATA_PATH directory.'''
    # loader = DirectoryLoader(DATA_PATH, glob='**/*.pdf')
    loader = DirectoryLoader('', glob='*.md')
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
        current_page_id = f'{source}:{page}'
        print(f'source: {source}, page: {page}')

        if current_page_id != last_page_id:
            current_chunk_index = 0
        else:
            current_chunk_index += 1
        last_page_id = current_page_id

        chunk.metadata['id'] = f'{current_page_id}:{current_chunk_index}'
    
    return chunks


def add_to_chroma(chunks):
    '''Add chunks to ChromaDB, skipping existing IDs.'''
    embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ.get('GEMINI_API_KEY'))
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection('results', embedding_function=embedding_fn)

    # Get existing IDs
    try:
        existing_ids = set(collection.get(include=[])['ids'])
    except Exception:
        existing_ids = set()

    # Only add new chunks
    new_chunks = [c for c in chunks if c.metadata['id'] not in existing_ids]

    if not new_chunks:
        print('✅ No new documents to add')
        return

    collection.add(
        documents=[c.page_content for c in new_chunks],
        metadatas=[c.metadata for c in new_chunks],
        ids=[c.metadata['id'] for c in new_chunks]
    )
    print(f'👉 Added {len(new_chunks)} new documents to ChromaDB at "{CHROMA_PATH}".')


def clean_chunks(chunks):
    ''' Clean the text chunks by removing images and non-printable characters.'''
    import re
    cleaned = []
    for c in chunks:
        text = c.page_content if hasattr(c, 'page_content') else str(c)
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
# chunks = clean_chunks(chunks)
print(f'Loaded {len(docs)} documents and split into {len(chunks)} chunks.')
add_to_chroma(chunks)
add_to_chroma(chunks)  # Run twice to test skipping existing
