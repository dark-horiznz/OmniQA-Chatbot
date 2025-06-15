from .qa_chains import create_base_chains, self_clarifying_qa, QA_chain_with_websearch
from src.templates import templates

import warnings
warnings.filterwarnings("ignore")

def make_chains():
    """
    Create and return the necessary chains for the application.
    Returns:
        dict: Contains the answer chain, clarify chain, summary chain,
              web summary chain, and final summary chain.
    """
    answer_template, clarify_template, summary_template, web_summary_template, final_summary_template = templates()
    answer_chain, clarify_chain, summary_chain, web_summary_chain, final_summary_chain = create_base_chains(
        answer_template,
        clarify_template,
        summary_template,
        web_summary_template,
        final_summary_template
    )

    return {
        "answer_chain": answer_chain,
        "clarify_chain": clarify_chain,
        "summary_chain": summary_chain,
        "web_summary_chain": web_summary_chain,
        "final_summary_chain": final_summary_chain
    }

def run_chains(user_question, vectorstore, max_queries=3, k_retrieval=3, web_mode=True):
    """
    Run the self-clarifying QA chains or the QA chain with web search based on the mode.
    
    Args:
        user_question (str): The question asked by the user.
        vectorstore (PineconeVectorStore): The vector store for similarity search.
        max_queries (int): Maximum number of queries to run.
        k_retrieval (int): Number of documents to retrieve.
        web_mode (bool): Whether to use web search mode or not.

    Returns:
        dict: The result of the QA chain execution.
    """
    chains = make_chains()
    if web_mode:
        return QA_chain_with_websearch(user_question, vectorstore, chains, max_queries, k_retrieval)
    else:
        return self_clarifying_qa(user_question, vectorstore, chains, max_queries, k_retrieval)['summary']