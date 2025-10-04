import google.generativeai as genai
from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content("Summarize this transcript and identify compliance issues.")
print(response.text)
