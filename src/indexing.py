import os
import chromadb

from sentence_transformers import SentenceTransformer

from .utils import hash_text


def retrieve_context_from_index(query: str, database_path: str, collection_name: str, 
                                embedding_model_identifier: str, top_k: int = 5) -> str:
    """
    Retrieve relevant context from the local Chroma database.

    Args:
        query (str): The index query.
        database_path (str): The path to the Chroma database.
        collection_name (str): The name of the collection to query.
        embedding_model_identifier (str): The model identifier for the sentence transformer.
        top_k (int): Number of top results to retrieve.

    Returns:
        str: Concatenated context string or None if no documents found.
    """
    # Ensure database path exists
    os.makedirs(database_path, exist_ok=True)

    # Initialize persistent client and load collection
    chroma_client = chromadb.PersistentClient(path=database_path)
    collection = chroma_client.get_or_create_collection(collection_name)

    # Check if the collection has any data
    if collection.count() == 0:
        return ""

    # Initialize embedding model
    embedder = SentenceTransformer(embedding_model_identifier)

    # Create query embedding
    query_emb = embedder.encode([query])

    # Query top_k most similar documents
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    # Extract documents safely
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    # Concatenate retrieved documents into a single string
    context = "\n\n".join(docs)

    return context

def save_context_to_index(context_with_sources: dict, database_path: str, collection_name: str, 
                          embedding_model_identifier: str):
    """
    Save context documents to the local Chroma database.

    Args:
        context_with_sources (dict): A dictionary mapping sources to their text content.
        database_path (str): The path to the Chroma database.
        collection_name (str): The name of the collection to save to.
        embedding_model_identifier (str): The model identifier for the sentence transformer.
    """
    # Ensure database path exists
    os.makedirs(database_path, exist_ok=True)

    # Initialize persistent client and load collection
    chroma_client = chromadb.PersistentClient(path=database_path)
    collection = chroma_client.get_or_create_collection(collection_name)

    # Initialize embedding model
    embedder = SentenceTransformer(embedding_model_identifier)

    # Prepare data for insertion
    texts = list(context_with_sources.values())
    ids = [hash_text(text) for text in texts]
    embeddings = embedder.encode(texts).tolist()

    # Add documents to the collection
    collection.add(documents=texts, ids=ids, embeddings=embeddings)
