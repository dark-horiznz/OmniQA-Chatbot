import google.generativeai as genai
from langchain_core.embeddings import Embeddings
import numpy as np

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"  # Updated to newer embedding model

    def embed_documents(self, texts):
        return [self._convert_to_float32(genai.embed_content(model=self.model_name, content=text, task_type="retrieval_document")["embedding"]) for text in texts]

    def embed_query(self, text):
        response = genai.embed_content(model=self.model_name, content=text, task_type="retrieval_query")
        return self._convert_to_float32(response["embedding"])

    @staticmethod
    def _convert_to_float32(embedding):
        return np.array(embedding, dtype=np.float32).tolist()