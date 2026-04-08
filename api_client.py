# api_client.py
# Central API client setup for GenAI Course
# Sprints 1–12

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# PRIMARY CLIENT — Groq (Sprints 1–8, 10–12)
# ============================================================

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model selector — import the right one per sprint
MODELS = {
    "fast":    "llama-3.1-8b-instant",      # Sprints 1, 2, 5
    "strong":  "llama-3.3-70b-versatile",   # Sprints 3, 6, 7, 8, 10, 11, 12
    "long":    "mixtral-8x7b-32768",         # Sprint 4
}

def get_groq_response(messages, model_type="fast", max_tokens=512):
    """
    Sends messages to Groq API.
    model_type: 'fast', 'strong', or 'long'
    """
    try:
        response = groq_client.chat.completions.create(
            model=MODELS[model_type],
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return "Sorry, I encountered an error."


# ============================================================
# VISION CLIENT — Google Gemini (Sprint 9 only)
# ============================================================

def get_gemini_vision_response(image_path, prompt):
    """
    Sends image + text prompt to Gemini Vision.
    Used in Sprint 9 only.
    """
    try:
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I encountered an error."


# ============================================================
# LOCAL CLIENT — Ollama (Sprint 2 SLM track)
# ============================================================

def get_ollama_response(messages, model="phi3"):
    """
    Sends messages to local Ollama SLM.
    Used in Sprint 2 for SLM comparison.
    Requires Ollama running locally: ollama serve
    """
    try:
        from ollama import Client
        client = Client()
        response = client.chat(
            model=model,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama error: {e}")
        return "Sorry, Ollama is not running. Run: ollama serve"


if __name__ == '__main__':
    # Quick test — confirms all clients are configured
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ]

    print("Testing Groq fast model...")
    print(get_groq_response(test_messages, model_type="fast"))

    print("\nTesting Groq strong model...")
    print(get_groq_response(test_messages, model_type="strong"))

    print("\nAll clients configured successfully.")