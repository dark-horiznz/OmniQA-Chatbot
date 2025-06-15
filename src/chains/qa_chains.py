from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain import LLMChain
from src.scraping import run

def create_base_chains(answer_template , clarify_template , summary_template , web_summary_template, final_summary_template):
    answer_chain = LLMChain(
    llm=ChatGroq(model="llama3-70b-8192"), 
    prompt=answer_template,
    )
    
    clarify_chain = LLMChain(
    llm=ChatGroq(model="llama3-70b-8192", temperature=0.2),
    prompt=clarify_template,
    )

    summary_chain = LLMChain(
    llm=ChatGroq(model="llama3-70b-8192"),
    prompt=summary_template,
    )
    
    web_summary_chain = LLMChain(
        llm=ChatGroq(model="llama3-70b-8192"),
        prompt=web_summary_template,
    )
    
    final_summary_chain = LLMChain(
        llm=ChatGroq(model="llama3-70b-8192"),
        prompt=final_summary_template,
    )
    return answer_chain , clarify_chain , summary_chain , web_summary_chain , final_summary_chain
    
def self_clarifying_qa(user_question: str,
                       vectorstore: PineconeVectorStore,
                       chains: dict,
                       max_queries: int = 3,
                       k_retrieval: int = 3):
    history = []
    q = user_question

    for depth in range(max_queries):
        docs = vectorstore.similarity_search(q, k=k_retrieval)
        ctx  = "\n\n".join(d.page_content for d in docs)

        a = chains['answer_chain'].run(question=q, context=ctx)
        history.append((q, a))

        if a.strip().lower() == "no suitable answer found in database.":
            break

        follow = chains['clarify_chain'].run(question=q, answer=a).strip()
        if follow.upper() == "ENOUGH":
            break

        q = follow

    hist_text = "\n".join(f"Q: {x}\nA: {y}" for x, y in history)
    summary = chains['summary_chain'].run(history=hist_text)

    return {"history": history, "summary": summary}

def QA_chain_with_websearch(user_question, vectorstore, chains, max_queries = 3, k_retrieval = 3 , web_mode = True): 
    result = self_clarifying_qa(user_question , vectorstore , chains, max_queries , k_retrieval)
    if web_mode:
        scrape_summary = run(user_question)
        print('Searching The web!')
        if scrape_summary:
            web_content = scrape_summary['web_content']
            web_summary = chains['web_summary_chain'].run(history=web_content)
            if result['summary'].strip().lower() == "no suitable answer found in database.":
                return web_summary
    try:
        final = chains['final_summary_chain'].run(web = web_summary , text = result['summary'])
        return final
    except:
        return result['summary']