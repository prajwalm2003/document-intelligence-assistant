# core/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE


def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
