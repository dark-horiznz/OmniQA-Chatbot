from .pc_utils import upsert , load_existing
from .postgres_utils import list_db_columns, fetch_qa_data
import warnings
warnings.filterwarnings("ignore")

def pinecone_db():
    return {
        "upsert": upsert,
        "load_existing": load_existing
    }

def postgres_db():
    return {
        "list_db_columns": list_db_columns,
        "fetch_qa_data": fetch_qa_data
    }

def __all__():
    return [
        pinecone_db,
        postgres_db
    ]