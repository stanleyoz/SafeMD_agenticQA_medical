"""
ingest_data.py - Knowledge Base Ingestion Script

This script reads PDF documents from the 'data/' folder and creates a vector
store index for semantic search. It's the first step in setting up the QA
system - run this ONCE before using qa_agent.py.

What it does:
    1. Loads all PDFs from the 'data/' directory
    2. Chunks the documents into searchable segments
    3. Creates vector embeddings using nomic-embed-text (via Ollama)
    4. Persists the index to 'storage/' for later use
Usage:
    1. Place your PDF files (e.g., NICE NG28 guidelines) in the 'data/' folder
    2. Run: python ingest_data.py
    3. The 'storage/' folder will be created with the vector index

Requirements:
    - Ollama running with nomic-embed-text and llama3.1:8b models pulled
    - PDF files in the 'data/' directory

Author: Stanley Chong
Project: MSc Computer Science, City University of London (DAM190)
"""

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ============================================================================
# CONFIGURATION: Set up embedding and LLM models
# ============================================================================
# nomic-embed-text: A compact embedding model (768 dimensions) that runs
# efficiently on consumer GPUs with 8GB VRAM.
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# llama3.1:8b: Used during indexing for any LLM-based processing
# request_timeout=360 allows for longer processing on slower hardware
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=360.0)


def ingest():
    """
    Main ingestion function - reads PDFs and creates a searchable vector index.

    The function performs three steps:
        1. Read: Load all PDF files from the 'data/' directory
        2. Embed: Convert text chunks into vector embeddings
        3. Persist: Save the index to 'storage/' for the QA agent to use

    The resulting index can be loaded by qa_agent.py for semantic retrieval.
    """
    print("Loading PDFs from 'data' folder...")

    # Step 1: Read all documents from the data directory
    # SimpleDirectoryReader handles PDF parsing automatically
    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()
    print(f"Found {len(documents)} pages.")

    # Step 2: Create the vector index
    # This chunks the documents and creates embeddings for each chunk
    # The embeddings are 768-dimensional vectors that capture semantic meaning
    print("Creating Vector Embeddings ...")
    index = VectorStoreIndex.from_documents(documents)

    # Step 3: Persist to disk so we don't have to re-embed every time
    # The storage folder contains the serialized index data
    print("Saving Index to 'storage'...")
    index.storage_context.persist(persist_dir="storage")
    print("Done! We can now run the Agent.")


if __name__ == "__main__":
    ingest()
