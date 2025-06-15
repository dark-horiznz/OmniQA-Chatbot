from .runner_service import configure_moudles , load_dataset_from_hf , load_data_to_vectorstore, load_existing_vectorstore, load_data_from_postgres
from src.chains import run_chains
import warnings
warnings.filterwarnings("ignore")

def initialize_resources(embeddings=None, vectorstore=None):
    """
    Initializes embeddings and vectorstore if not provided.
    """
    if not embeddings or not vectorstore:
        embeddings, _ = configure_moudles()
    return embeddings, vectorstore

def load_data(
    hf_dataset_name=None,
    postgres_table=None,
    postgres_query_col=None,
    postgres_answer_col=None,
    hf_query_col=None,
    hf_answer_col=None,
    subset=None,
    split='train',
    shuffle=True
):
    """
    Loads data either from Hugging Face or Postgres, depending on provided arguments.
    """
    if hf_dataset_name and hf_query_col and hf_answer_col:
        return load_dataset_from_hf(
            hf_dataset_name, hf_query_col, hf_answer_col, subset, split, shuffle
        )

    if postgres_table and postgres_query_col and postgres_answer_col:
        return load_data_from_postgres(
            postgres_query_col, postgres_answer_col, postgres_table
        )

    return None

def select_vectorstore(
    hf_dataset_name=None,
    postgres_table=None,
    postgres_query_col=None,
    postgres_answer_col=None,
    hf_query_col=None,
    hf_answer_col=None,
    subset=None,
    split='train',
    shuffle=True,
    vectorstore=None,
    embeddings=None
):
    """
    Selects or creates the appropriate vectorstore.
    """
    embeddings, vectorstore = initialize_resources(embeddings, vectorstore)

    df = load_data(
        hf_dataset_name, postgres_table, postgres_query_col, postgres_answer_col,
        hf_query_col, hf_answer_col, subset, split, shuffle
    )

    if df is not None:
        if not vectorstore:
            vectorstore = load_data_to_vectorstore(df, embeddings)
    elif not vectorstore:
        vectorstore = load_existing_vectorstore(embeddings)

    return vectorstore

def run_pipeline(
    question,
    vectorstore,
    max_queries=5,
    k_retrieval=5,
    web_mode=True
):
    """
    Runs the entire pipeline for question answering.
    """
    return run_chains(question, vectorstore, max_queries, k_retrieval, web_mode)


def run_main(
    question,
    hf_dataset_name=None,
    postgres_table=None,
    postgres_query_col=None,
    postgres_answer_col=None,
    hf_query_col=None,
    hf_answer_col=None,
    subset=None,
    split='train',
    shuffle=True,
    vectorstore=None,
    embeddings=None,
    max_queries=5,
    k_retrieval=5,
    web_mode=True
):
    """
    Main entry point to run the pipeline with full configuration.
    """
    vectorstore = select_vectorstore(
        hf_dataset_name, postgres_table, postgres_query_col, postgres_answer_col,
        hf_query_col, hf_answer_col, subset, split, shuffle, vectorstore, embeddings
    )

    return run_pipeline(question, vectorstore, max_queries, k_retrieval, web_mode)



