from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate

def make_templates():
    answer_template = ChatPromptTemplate.from_template(
                """
            You are a helpful assistant.
            
            Context:
            {context}
            
            User question:
            {question}
            
            Task: Provide a concise, accurate answer based solely on the context.
            If the context does NOT contain the information needed to answer the question, reply exactly:
            "No suitable answer found in database."
            """
            )

    clarify_template = ChatPromptTemplate.from_template(
        """
    You answered:
    {answer}
    
    The user originally asked:
    {question}
    
    Task: Determine if more information is needed to answer correctly.
    - If you CANNOT answer because the database lacked relevant information, reply exactly `ENOUGH`.
    - If you NEED more info, ask a single, specific follow-up question.
    - If you have enough context to be confident, reply exactly `ENOUGH`.
    """
    )
    
    summary_template = ChatPromptTemplate.from_template(
        """
    Here are the question/answer pairs that occurred:
    {history}
    
    Task: Provide a concise final summary of the information above.
    If all answers were 'No suitable answer found in database.', reply:
    "No information available to summarize."
    """
    )
    
    web_summary_template = ChatPromptTemplate.from_template(
        """
    Here is the webcontext we have scraped it from internet:
    {history}
    
    Task: Provide a concise final summary of the information above.
    If all answers are not relevent simply reply 'No relevent information found on the web'.
    If relevent content is found, summarise the content properly and properly cite it as web search content.
    """
    )
    
    final_summary_template = ChatPromptTemplate.from_template(
        """
    Here is the webcontext we have scraped it from internet:
    {web}
    Below is the context we have found from the corpus text:
    {text}
    
    Task: Provide a concise and detailed final summary of the information above.
    If all answers are not relevent simply reply 'No relevent information Is available please contact support'.
    If relevent content is found, summarise the content properly and properly cite content, give more priority to corpus text.
    If corpus text is less relevent then give priority to web content.
    Properly cite web content in the final summary. Make it sure the user knows the additional answer is from the web.
    """
    )
    return answer_template , clarify_template , summary_template , web_summary_template, final_summary_template