from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from services.openrouter_service import OpenRouterService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
openrouter_service = OpenRouterService()

router = APIRouter()

class TranslationRequest(BaseModel):
    content: str
    target_language: str = "ur"  # Default to Urdu
    source_language: str = "en"  # Default from English

class TranslationResponse(BaseModel):
    original_content: str
    translated_content: str
    source_language: str
    target_language: str

class TranslationBatchRequest(BaseModel):
    texts: List[str]
    source_lang: str = "en"
    target_lang: str = "ur"

class TranslationBatchResponse(BaseModel):
    original_texts: List[str]
    translated_texts: List[str]
    source_language: str
    target_language: str

class CacheStatusResponse(BaseModel):
    total_cached: int
    cache_size_mb: float
    oldest_entry: Optional[str]
    newest_entry: Optional[str]

# Translation cache for performance optimization
# Simple in-memory cache (in production, you'd use Redis or similar)
translation_cache: Dict[str, Dict[str, Any]] = {}

@router.post("/urdu", response_model=TranslationResponse)
async def translate_to_urdu(translation_data: TranslationRequest):
    """
    Translate content to Urdu using OpenRouter
    """
    try:
        logger.info(f"Translating content to {translation_data.target_language}")

        # Create a cache key
        cache_key = f"{translation_data.source_language}_{translation_data.target_language}_{translation_data.content}"

        # Check if translation is already cached
        if cache_key in translation_cache:
            cached_result = translation_cache[cache_key]
            logger.info(f"Returning cached translation for key: {cache_key[:50]}...")
            return TranslationResponse(
                original_content=translation_data.content,
                translated_content=cached_result['translated_content'],
                source_language=translation_data.source_language,
                target_language=translation_data.target_language
            )

        # For translation, we'll use OpenRouter's generation capabilities
        # In a real implementation, you might use a dedicated translation model
        prompt = f"""
        Translate the following text to {translation_data.target_language}:

        {translation_data.content}

        Please provide only the translated text without any additional commentary.
        """

        # Note: For actual translation, OpenRouter's model supports multiple languages
        # using the language model with a translation prompt
        translated_content = await openrouter_service.generate_response(prompt)

        # Cache the result
        translation_cache[cache_key] = {
            'translated_content': translated_content,
            'timestamp': datetime.now(),
            'source_language': translation_data.source_language,
            'target_language': translation_data.target_language
        }

        return TranslationResponse(
            original_content=translation_data.content,
            translated_content=translated_content,
            source_language=translation_data.source_language,
            target_language=translation_data.target_language
        )
    except Exception as e:
        logger.error(f"Error translating content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate", response_model=TranslationResponse)
async def translate_content(translation_data: TranslationRequest):
    """
    Translate content between languages using OpenRouter
    """
    try:
        logger.info(f"Translating from {translation_data.source_language} to {translation_data.target_language}")

        # Create a cache key
        cache_key = f"{translation_data.source_language}_{translation_data.target_language}_{translation_data.content}"

        # Check if translation is already cached
        if cache_key in translation_cache:
            cached_result = translation_cache[cache_key]
            logger.info(f"Returning cached translation for key: {cache_key[:50]}...")
            return TranslationResponse(
                original_content=translation_data.content,
                translated_content=cached_result['translated_content'],
                source_language=translation_data.source_language,
                target_language=translation_data.target_language
            )

        # Create a translation prompt
        prompt = f"""
        Translate the following text from {translation_data.source_language} to {translation_data.target_language}:

        {translation_data.content}

        Please provide only the translated text without any additional commentary.
        """

        translated_content = await openrouter_service.generate_response(prompt)

        # Cache the result
        translation_cache[cache_key] = {
            'translated_content': translated_content,
            'timestamp': datetime.now(),
            'source_language': translation_data.source_language,
            'target_language': translation_data.target_language
        }

        return TranslationResponse(
            original_content=translation_data.content,
            translated_content=translated_content,
            source_language=translation_data.source_language,
            target_language=translation_data.target_language
        )
    except Exception as e:
        logger.error(f"Error translating content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/urdu-batch", response_model=TranslationBatchResponse)
async def translate_batch_to_urdu(translation_data: TranslationBatchRequest):
    """
    Translate multiple texts to Urdu in batch
    """
    try:
        logger.info(f"Translating {len(translation_data.texts)} texts to {translation_data.target_lang}")

        translated_texts = []
        for text in translation_data.texts:
            # Create a cache key for each text
            cache_key = f"{translation_data.source_lang}_{translation_data.target_lang}_{text}"

            # Check if translation is already cached
            if cache_key in translation_cache:
                cached_result = translation_cache[cache_key]
                translated_texts.append(cached_result['translated_content'])
                logger.info(f"Returning cached translation for key: {cache_key[:30]}...")
            else:
                # Create a translation prompt
                prompt = f"""
                Translate the following text from {translation_data.source_lang} to {translation_data.target_lang}:

                {text}

                Please provide only the translated text without any additional commentary.
                """

                translated_text = await openrouter_service.generate_response(prompt)

                # Cache the result
                translation_cache[cache_key] = {
                    'translated_content': translated_text,
                    'timestamp': datetime.now(),
                    'source_language': translation_data.source_lang,
                    'target_language': translation_data.target_lang
                }

                translated_texts.append(translated_text)

        return TranslationBatchResponse(
            original_texts=translation_data.texts,
            translated_texts=translated_texts,
            source_language=translation_data.source_lang,
            target_language=translation_data.target_lang
        )
    except Exception as e:
        logger.error(f"Error translating batch content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-status", response_model=CacheStatusResponse)
async def get_cache_status():
    """
    Get translation cache status
    """
    if not translation_cache:
        return CacheStatusResponse(
            total_cached=0,
            cache_size_mb=0.0,
            oldest_entry=None,
            newest_entry=None
        )

    # Calculate cache statistics
    total_cached = len(translation_cache)

    # Approximate cache size (in production, you'd calculate actual memory usage)
    cache_size_mb = len(str(translation_cache)) / (1024 * 1024)

    # Find oldest and newest entries
    timestamps = [entry['timestamp'] for entry in translation_cache.values()]
    oldest = min(timestamps) if timestamps else None
    newest = max(timestamps) if timestamps else None

    return CacheStatusResponse(
        total_cached=total_cached,
        cache_size_mb=round(cache_size_mb, 2),
        oldest_entry=oldest.isoformat() if oldest else None,
        newest_entry=newest.isoformat() if newest else None
    )

@router.post("/clear-cache")
async def clear_translation_cache():
    """
    Clear the translation cache
    """
    global translation_cache
    count = len(translation_cache)
    translation_cache.clear()

    logger.info(f"Cleared {count} entries from translation cache")

    return {
        "message": f"Successfully cleared {count} cached translations",
        "cleared_count": count
    }

@router.get("/supported-languages")
async def supported_languages():
    """
    Get list of supported languages for translation
    """
    # Based on OpenRouter's multilingual model support
    supported = [
        {"code": "en", "name": "English"},
        {"code": "ur", "name": "Urdu"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "ar", "name": "Arabic"}
    ]

    return {
        "supported_languages": supported,
        "default_target": "ur"
    }