import { useState, useCallback } from 'react';

interface TranslationRequest {
  text: string;
  source_lang?: string;
  target_lang?: string;
}

interface TranslationResponse {
  original_text: string;
  translated_text: string;
  source_language: string;
  target_language: string;
}

interface TranslationBatchRequest {
  texts: string[];
  source_lang?: string;
  target_lang?: string;
}

interface TranslationBatchResponse {
  original_texts: string[];
  translated_texts: string[];
  source_language: string;
  target_language: string;
}

export function useTranslation(backendUrl: string = 'http://localhost:8000') {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const translateToUrdu = useCallback(async (text: string): Promise<TranslationResponse> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/translate/urdu`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: text,
          source_language: 'en',
          target_language: 'ur',
        }),
      });

      if (!response.ok) {
        throw new Error(`Translation API error: ${response.status} ${response.statusText}`);
      }

      const result: TranslationResponse = await response.json();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl]);

  const translateBatchToUrdu = useCallback(async (texts: string[]): Promise<TranslationBatchResponse> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/translate/urdu-batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          texts,
          source_lang: 'en',
          target_lang: 'ur',
        }),
      });

      if (!response.ok) {
        throw new Error(`Batch translation API error: ${response.status} ${response.statusText}`);
      }

      const result: TranslationBatchResponse = await response.json();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl]);

  const getCacheStatus = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/translate/cache-status`);

      if (!response.ok) {
        throw new Error(`Cache status API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    }
  }, [backendUrl]);

  const clearCache = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/translate/clear-cache`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Clear cache API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    }
  }, [backendUrl]);

  return {
    translateToUrdu,
    translateBatchToUrdu,
    getCacheStatus,
    clearCache,
    isLoading,
    error,
  };
}