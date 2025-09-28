import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import os , sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embeddings import get_embedding_models

def init_faiss(embeddings_dim = 612):
    """Initialize and return a FAISS index."""
    index = faiss.IndexFlatL2(embeddings_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store

def upsert(docs, embeddings, vectorstore=None):
    """
    Upsert documents into the FAISS vector store.
    Args:
        docs (List[Document]): List of documents to be upserted.
        embeddings (Embeddings): The embeddings model to use for vectorization.
        vectorstore (FAISS, optional): Existing FAISS vectorstore to add to.
    Returns:
        FAISS: The vector store containing the upserted documents.
    """
    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs, ids=[str(uuid4()) for _ in docs])
    return vectorstore

def load_existing(embeddings, faiss_index_path="faiss_index"):
    """
    Load an existing FAISS index from disk.
    Args:
        embeddings (Embeddings): The embeddings model to use for vectorization.
        faiss_index_path (str): Path to the FAISS index file.
    Returns:
        FAISS: The loaded FAISS vector store.
    """
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        print(f"FAISS index not found at {faiss_index_path}, creating new empty vector store")
        return None

def save_vectorstore(vectorstore, faiss_index_path="faiss_index"):
    """
    Save FAISS vector store to disk.
    Args:
        vectorstore (FAISS): The FAISS vector store to save.
        faiss_index_path (str): Path where to save the FAISS index.
    """
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")
    
if __name__ == "__main__":
    # Example usage
    embeddings = get_embedding_models()['GeminiEmbeddings'](api_key=os.environ["GEMINI_API_KEY"])
    vectorstore = init_faiss(embeddings)
    print("FAISS index initialized.")

