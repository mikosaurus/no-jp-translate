#!/usr/bin/env python3
"""
Script to download and cache translation models during Docker build.
This ensures models are available immediately when the container starts.
"""

from transformers import MarianMTModel, MarianTokenizer
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models to download
MODELS = {
    "no-en": "Helsinki-NLP/opus-mt-gmq-en",
    "en-no": "Helsinki-NLP/opus-mt-en-gmq"
}

def download_model(model_key: str, model_name: str) -> bool:
    """Download and cache a single model"""
    try:
        logger.info(f"Downloading {model_name}...")
        
        # Download tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer for {model_name} downloaded")
        
        # Download model
        model = MarianMTModel.from_pretrained(model_name)
        logger.info(f"✓ Model {model_name} downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False

def main():
    """Download all models"""
    logger.info("Starting model download process...")
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_key, model_name in MODELS.items():
        if download_model(model_key, model_name):
            success_count += 1
        else:
            logger.error(f"Failed to download {model_key}: {model_name}")
    
    if success_count == total_count:
        logger.info(f"✓ All {total_count} models downloaded successfully!")
        return 0
    else:
        logger.error(f"✗ Only {success_count}/{total_count} models downloaded successfully")
        return 1

if __name__ == "__main__":
    sys.exit(main())
