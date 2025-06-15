from .pc_utils import upsert , load_existing
import warnings
warnings.filterwarnings("ignore")

def pinecone_db():
    return {
        "upsert": upsert,
        "load_existing": load_existing
    }