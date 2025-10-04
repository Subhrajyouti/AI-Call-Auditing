import google.generativeai as genai
from app.config import GEMINI_API_KEY

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def transcribe_audio(file_path: str):
    """
    Transcribes audio using Gemini 1.5 Pro model.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Read the audio file as binary
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Gemini supports multi-modal input (audio + text prompt)
    response = model.generate_content([
        {"mime_type": "audio/mp3", "data": audio_bytes},
        {"text": "Transcribe this call conversation clearly with speaker labels if possible."}
    ])

    # Return the text
    return response.text
