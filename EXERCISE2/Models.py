from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

fast_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=1,
)

powerful_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=1,
    disable_streaming=True,
)

secondary_model = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_retries=1,
    disable_streaming=True,
)