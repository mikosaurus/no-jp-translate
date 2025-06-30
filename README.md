# Install dependencies
pip install -r requirements.txt

# Run the server
python translation_api.py

# The API will be available at:
# http://localhost:8000
# API documentation: http://localhost:8000/docs

# --- Usage Examples ---

# 1. Norwegian to Japanese
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hei, hvordan har du det?"}'

# 2. English to Japanese  
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'

# 3. Japanese to Norwegian
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは、元気ですか？"}'

# 4. With explicit source language
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "God morgen", "source_language": "norwegian"}'

# 5. Preload all models (recommended on startup)
curl -X GET "http://localhost:8000/models/preload"

# 6. Health check
curl -X GET "http://localhost:8000/health"

# --- Python client example ---
import requests

def translate_text(text, source_language=None):
    url = "http://localhost:8000/translate"
    payload = {"text": text}
    if source_language:
        payload["source_language"] = source_language
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
result = translate_text("Hei, hvordan har du det?")
print(f"Translation: {result['translated_text']}")
print(f"Path: {result['translation_path']}"
