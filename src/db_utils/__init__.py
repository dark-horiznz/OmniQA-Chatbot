from .faiss_utils import upsert as faiss_upsert, load_existing as faiss_load_existing, save_vectorstore as faiss_save
from .pc_utils import upsert as pc_upsert, load_existing as pc_load_existing
from .postgres_utils import list_db_columns, fetch_qa_data
import warnings
warnings.filterwarnings("ignore")

def faiss_db():
    """FAISS vector database utilities"""
    return {
        "upsert": faiss_upsert,
        "load_existing": faiss_load_existing,
        "save": faiss_save
    }

def pinecone_db():
    """Pinecone vector database utilities (legacy)"""
    return {
        "upsert": pc_upsert,
        "load_existing": pc_load_existing
    }

def postgres_db():
    """PostgreSQL database utilities"""
    return {
        "list_db_columns": list_db_columns,
        "fetch_qa_data": fetch_qa_data
    }

def __all__():
    return [
        faiss_db,
        pinecone_db,
        postgres_db
    ]