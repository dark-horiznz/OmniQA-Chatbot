import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from lxml import html
import time
import random
from tqdm import tqdm
from typing import List

def get_gemini_model(model_name = "gemini-1.5-pro", temperature = 0.4):
    return genai.GenerativeModel(model_name)

def get_generation_config(temperature = 0.4):
    return {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

def get_safety_settings():
    return [
        {"category": category, "threshold": "BLOCK_NONE"}
        for category in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]
    
def generate_gemini_response(model, prompt):
    response = model.generate_content(
        prompt,
        generation_config=get_generation_config(),
        safety_settings=get_safety_settings()
    )
    if response.candidates and len(response.candidates) > 0:
        return response.candidates[0].content.parts[0].text
    return ''

def create_search_prompt(query, context = ""):
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    Consider the provided context if available. 
    If the query requires external information, No context is provided, Irrelevent context is present or latest information is required, then output the special token <SEARCH> 
    followed by relevant keywords extracted from the query to optimize for search engine results. 
    Ensure the keywords are concise and relevant. If document data is sufficient, simply return blank."""
    
    if context:
        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"
    
    return f"{system_prompt}\n\nQuery: {query}"

def create_summary_prompt(content):
    return f"""Please provide a comprehensive yet concise summary of the following content, highlighting the most important points and maintaining factual accuracy. Organize the information in a clear and coherent manner:

Content to summarize:
{content}

Summary:"""

def init_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def extract_static_page(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None
        
def extract_dynamic_page(url, driver):
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        
        body = driver.find_element(By.TAG_NAME, "body")
        ActionChains(driver).move_to_element(body).perform()
        time.sleep(random.uniform(2, 5))
        
        page_source = driver.page_source
        tree = html.fromstring(page_source)
        
        text = tree.xpath('//body//text()')
        text_content = ' '.join(text).strip()
        return text_content[:1000]

    except Exception as e:
        print(f"Error fetching dynamic page: {e}")
        return None

def scrape_page(url):
    if "javascript" in url or "dynamic" in url:
        driver = init_selenium_driver()
        text = extract_dynamic_page(url, driver)
        driver.quit()
    else:
        text = extract_static_page(url)
    
    return text

def scrape_web(urls, max_urls = 5):
    texts = []
    
    for url in tqdm(urls[:max_urls], desc="Scraping websites"):
        text = scrape_page(url)
        
        if text:
            texts.append(text)
        else:
            print(f"Failed to retrieve content from {url}")
            
    return texts

def check_search_needed(model, query , context):
    prompt = create_search_prompt(query , context)
    response = generate_gemini_response(model, prompt)
    
    if "<SEARCH>" in response:
        search_terms = response.split("<SEARCH>")[1].strip()
        return True, search_terms
    return False, None

def summarize_content(model, content):
    prompt = create_summary_prompt(content)
    return generate_gemini_response(model, prompt)

def process_query(query , context = ''):
    model = get_gemini_model()
    search_tool = DuckDuckGoSearchRun()
    
    needs_search, search_terms = check_search_needed(model, query , context)
    
    result = {
        "original_query": query,
        "needs_search": needs_search,
        "search_terms": search_terms,
        "web_content": None,
        "summary": None,
        "raw_response": None
    }
    
    if needs_search:
        search_results = search_tool.run(search_terms)
        result["web_content"] = search_results
        
        summary = summarize_content(model, search_results)
        result["summary"] = summary
    
    return result

def run(question , context = '' , debug = False):
    result = process_query(question , context = '')
    if debug:
        print("\nQuery Results:")
        print(f"Search needed: {result['needs_search']}")
        
        if result['needs_search']:
            print(f"\nSearch terms used: {result['search_terms']}")
            print("\nSummary of findings:")
            print(result['summary'])
    return result