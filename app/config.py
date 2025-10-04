import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
