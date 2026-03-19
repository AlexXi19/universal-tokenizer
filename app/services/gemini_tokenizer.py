import warnings

from google.genai._common import ExperimentalWarning
from google.genai.local_tokenizer import LocalTokenizer
from app.services.base_tokenizer import BaseTokenizer
from app.services.logger import logger


class GeminiTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = LocalTokenizer(model_name=model_name)
        logger.info(f"[GeminiTokenizer] Loaded LocalTokenizer for: {model_name}")

    def count_tokens(self, text: str) -> dict:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
            result = self.tokenizer.count_tokens(text)
        return {
            "token_count": result.total_tokens,
            "model": self.model_name,
            "tokenizer": "gemini",
        }
