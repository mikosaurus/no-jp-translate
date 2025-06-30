from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
from typing import Optional
import langdetect
from langdetect import detect
import logging
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Translation API", description="Norwegian/English/Japanese Translation Service")

# Initialize Google Translator
google_translator = Translator()

# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    source_language: Optional[str] = None  # Optional language detection

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    translation_path: str

# Translation models configuration - Only for Norwegian/English
MODELS = {
    "no-en": "Helsinki-NLP/opus-mt-gmq-en",      # North Germanic to English (includes Norwegian)
    "en-no": "Helsinki-NLP/opus-mt-en-gmq"       # English to North Germanic (includes Norwegian)
}

# Global model cache
model_cache = {}
tokenizer_cache = {}

def load_model(model_key: str):
    """Load and cache translation model"""
    if model_key not in model_cache:
        logger.info(f"Loading model: {MODELS[model_key]}")
        try:
            tokenizer_cache[model_key] = MarianTokenizer.from_pretrained(MODELS[model_key])
            model_cache[model_key] = MarianMTModel.from_pretrained(MODELS[model_key])
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model_cache[model_key] = model_cache[model_key].cuda()
                logger.info(f"Model {model_key} loaded on GPU")
            else:
                logger.info(f"Model {model_key} loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load translation model: {model_key}")
    
    return tokenizer_cache[model_key], model_cache[model_key]

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        detected = detect(text)
        # Map langdetect codes to our system
        lang_mapping = {
            'no': 'norwegian',
            'nb': 'norwegian',  # Norwegian Bokmål
            'nn': 'norwegian',  # Norwegian Nynorsk
            'da': 'norwegian',  # Danish often confused with Norwegian - treat as Norwegian
            'en': 'english',
            'ja': 'japanese'
        }
        return lang_mapping.get(detected, detected)
    except:
        logger.warning(f"Language detection failed for text: {text[:50]}...")
        return "unknown"

def translate_with_google(text: str, source_lang: str, target_lang: str) -> str:
    """Translate using Google Translate for Japanese translations"""
    try:
        # Map our language codes to Google's
        google_lang_map = {
            'english': 'en',
            'japanese': 'ja',
            'norwegian': 'no'
        }
        
        src = google_lang_map.get(source_lang, source_lang)
        tgt = google_lang_map.get(target_lang, target_lang)
        
        result = google_translator.translate(text, src=src, dest=tgt)
        return result.text
        
    except Exception as e:
        logger.error(f"Google Translate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Google translation failed: {str(e)}")

def translate_text(text: str, model_key: str, target_lang: str = None) -> str:
    """Translate text using specified model"""
    tokenizer, model = load_model(model_key)
    
    # Add language token for group models that need it
    if model_key == "en-no":
        # English to Norwegian - add Norwegian target token  
        text = ">>nb<< " + text  # Use Norwegian Bokmål
    
    # Tokenize and translate
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to same device as model
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate translation with better parameters for natural output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=512, 
            num_beams=5,              # More beams for better quality
            length_penalty=1.0,       # Neutral length preference  
            early_stopping=True,
            do_sample=True,           # Add sampling for more natural output
            temperature=0.8,          # Some randomness for naturalness
            top_p=0.9                 # Nucleus sampling
        )
    
    # Decode output
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

def get_translation_path(source_lang: str, target_lang: str) -> tuple:
    """Determine translation path and required models/services"""
    
    # Norwegian to Japanese (Helsinki-NLP + Google)
    if source_lang == "norwegian" and target_lang == "japanese":
        return [("helsinki", "no-en"), ("google", "english", "japanese")], "norwegian → english → japanese"
    
    # English to Japanese (Google direct)
    elif source_lang == "english" and target_lang == "japanese":
        return [("google", "english", "japanese")], "english → japanese"
    
    # Japanese to Norwegian (Google + Helsinki-NLP)
    elif source_lang == "japanese" and target_lang == "norwegian":
        return [("google", "japanese", "english"), ("helsinki", "en-no")], "japanese → english → norwegian"
    
    else:
        raise HTTPException(status_code=400, detail=f"Translation path not supported: {source_lang} → {target_lang}")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate text based on the three supported cases"""
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Detect language if not provided
    if request.source_language:
        source_lang = request.source_language.lower()
    else:
        source_lang = detect_language(text)
        logger.info(f"Detected language: {source_lang}")
    
    # Determine target language and translation path
    if source_lang in ["norwegian", "english"]:
        target_lang = "japanese"
    elif source_lang == "japanese":
        target_lang = "norwegian"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {source_lang}")
    
    try:
        # Get translation path
        model_sequence, path_description = get_translation_path(source_lang, target_lang)
        
        # Perform translation(s) using hybrid approach
        current_text = text
        for step in model_sequence:
            if step[0] == "helsinki":
                # Use Helsinki-NLP model
                model_key = step[1]
                logger.info(f"Translating with Helsinki-NLP model: {model_key}")
                current_text = translate_text(current_text, model_key, target_lang)
            elif step[0] == "google":
                # Use Google Translate
                src_lang = step[1]
                tgt_lang = step[2]
                logger.info(f"Translating with Google Translate: {src_lang} → {tgt_lang}")
                current_text = translate_with_google(current_text, src_lang, tgt_lang)
            
            logger.info(f"Intermediate result: {current_text[:100]}...")
        
        return TranslationResponse(
            original_text=text,
            translated_text=current_text,
            source_language=source_lang,
            target_language=target_lang,
            translation_path=path_description
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": list(model_cache.keys())}

@app.get("/models/preload")
async def preload_models():
    """Preload Helsinki-NLP translation models"""
    loaded_models = []
    failed_models = []
    
    for model_key in MODELS.keys():
        try:
            load_model(model_key)
            loaded_models.append(model_key)
        except Exception as e:
            failed_models.append({"model": model_key, "error": str(e)})
    
    # Test Google Translate
    try:
        google_translator.translate("test", src='en', dest='ja')
        loaded_models.append("google-translate")
    except Exception as e:
        failed_models.append({"model": "google-translate", "error": str(e)})
    
    return {
        "loaded_models": loaded_models,
        "failed_models": failed_models,
        "total_loaded": len(loaded_models),
        "translation_approach": "Hybrid: Helsinki-NLP for Norwegian ↔ English, Google Translate for Japanese"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
