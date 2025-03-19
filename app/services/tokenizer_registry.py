from app.services.huggingface_tokenizer import HuggingFaceTokenizer
from app.services.openai_tokenizer import OpenAITokenizer
import tiktoken
from app.services.logger import logger
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Optional

DEFAULT_TOKENIZER = 'o200k_base'


class TokenizerRegistry:
    def __init__(self, preload_tokenizers=None):
        logger.info("[TokenizerRegistry] Initializing TokenizerRegistry")
        self.tokenizers = {}
        self._tokenizer_type_cache = {}
        self._failed_tokenizers = set()  # Cache for failed tokenizer attempts
        self._loading_tokenizers = {}  # Track tokenizers being loaded
        self._executor = ThreadPoolExecutor(max_workers=3)  # Background thread pool
        self._lock = threading.Lock()

        # Ensure default tokenizer is loaded first
        logger.info(f"[TokenizerRegistry] Loading default tokenizer: {DEFAULT_TOKENIZER}")
        self._ensure_default_tokenizer()

        # Then load any additional preload tokenizers
        if preload_tokenizers:
            logger.info(f"[TokenizerRegistry] Preloading tokenizers: {preload_tokenizers}")
            self._preload_tokenizers(preload_tokenizers)

    def _preload_tokenizers(self, preload_tokenizers):
        for model_name in preload_tokenizers:
            self.register_tokenizer(model_name)

    def get_tokenizer_type(self, model_name: str):
        logger.debug(f"[TokenizerRegistry] Determining tokenizer type for model: {model_name}")
        if model_name in self._tokenizer_type_cache:
            logger.debug(f"[TokenizerRegistry] Found cached tokenizer type for {model_name}: {self._tokenizer_type_cache[model_name]}")
            return self._tokenizer_type_cache[model_name]

        try:
            # Try OpenAI tokenizer first
            logger.debug(f"[TokenizerRegistry] Attempting to load OpenAI tokenizer for {model_name}")
            tiktoken.encoding_for_model(model_name) or tiktoken.get_encoding(model_name)
            tokenizer_type = "openai"
            logger.debug(f"[TokenizerRegistry] Successfully loaded OpenAI tokenizer for {model_name}")
        except (KeyError, ValueError) as e:
            logger.debug(f"[TokenizerRegistry] OpenAI tokenizer failed for {model_name}, trying HuggingFace. Error: {str(e)}")
            try:
                # Try HuggingFace tokenizer as fallback
                HuggingFaceTokenizer(model_name)
                tokenizer_type = "huggingface"
                logger.debug(f"[TokenizerRegistry] Successfully loaded HuggingFace tokenizer for {model_name}")
            except OSError as e:
                # Default to OpenAI if both fail
                logger.warning(f"[TokenizerRegistry] Both tokenizer types failed for {model_name}, defaulting to OpenAI. Error: {str(e)}")
                tokenizer_type = "openai"

        self._tokenizer_type_cache[model_name] = tokenizer_type
        logger.debug(f"[TokenizerRegistry] Caching tokenizer type for {model_name}: {tokenizer_type}")
        return tokenizer_type

    def _async_register_tokenizer(self, model_name: str) -> None:
        logger.info(f"[TokenizerRegistry] Starting async registration of tokenizer: {model_name}")
        try:
            tokenizer_type = self.get_tokenizer_type(model_name)
            logger.debug(f"[TokenizerRegistry] Creating {tokenizer_type} tokenizer for {model_name}")
            if tokenizer_type == "huggingface":
                tokenizer = HuggingFaceTokenizer(model_name)
            else:
                tokenizer = OpenAITokenizer(model_name)
            
            with self._lock:
                logger.debug(f"[TokenizerRegistry] Acquired lock, storing tokenizer for {model_name}")
                self.tokenizers[model_name] = tokenizer
                self._loading_tokenizers.pop(model_name, None)
            logger.info(f"[TokenizerRegistry] Tokenizer registered asynchronously: {model_name}")
        except Exception as e:
            with self._lock:
                logger.error(f"[TokenizerRegistry] Failed to load tokenizer {model_name} asynchronously: {str(e)}")
                self._failed_tokenizers.add(model_name)
                self._loading_tokenizers.pop(model_name, None)
            logger.warning(f"[TokenizerRegistry] Failed to load tokenizer {model_name}: {str(e)}")

    def register_tokenizer(self, model_name: str):
        logger.info(f"[TokenizerRegistry] Attempting to register tokenizer: {model_name}")
        if model_name in self._failed_tokenizers:
            logger.warning(f"[TokenizerRegistry] Skipping previously failed tokenizer: {model_name}")
            return

        tokenizer_type = self.get_tokenizer_type(model_name)
        logger.debug(f"[TokenizerRegistry] Creating {tokenizer_type} tokenizer for {model_name}")
        if tokenizer_type == "huggingface":
            self.tokenizers[model_name] = HuggingFaceTokenizer(model_name)
        elif tokenizer_type == "openai":
            self.tokenizers[model_name] = OpenAITokenizer(model_name)
        else:
            logger.warning(f"[TokenizerRegistry] Unknown tokenizer type {tokenizer_type}, using default tokenizer")
            self.tokenizers[model_name] = OpenAITokenizer(DEFAULT_TOKENIZER)

        logger.info(f"[TokenizerRegistry] Tokenizer registered: {model_name}")

    def get_tokenizer(self, model_name: str):
        logger.debug(f"[TokenizerRegistry] Requesting tokenizer for model: {model_name}")
        # If we know this tokenizer failed before, immediately use default
        if model_name in self._failed_tokenizers:
            logger.debug(f"[TokenizerRegistry] Using default tokenizer due to previous failure of {model_name}")
            return self.tokenizers.get(DEFAULT_TOKENIZER) or self._ensure_default_tokenizer()

        # Return existing tokenizer if available
        if model_name in self.tokenizers:
            logger.debug(f"[TokenizerRegistry] Found existing tokenizer for {model_name}")
            return self.tokenizers[model_name]

        # If tokenizer is currently loading, use default
        if model_name in self._loading_tokenizers:
            logger.debug(f"[TokenizerRegistry] Tokenizer {model_name} is currently loading, using default")
            return self.tokenizers.get(DEFAULT_TOKENIZER) or self._ensure_default_tokenizer()

        # Try to load the requested tokenizer
        try:
            # Start background loading if not already loading
            if model_name not in self._loading_tokenizers:
                logger.debug(f"[TokenizerRegistry] Starting background loading for tokenizer {model_name}")
                self._loading_tokenizers[model_name] = self._executor.submit(
                    self._async_register_tokenizer, model_name
                )
            # Return default tokenizer while loading
            logger.debug(f"[TokenizerRegistry] Returning default tokenizer while {model_name} loads")
            return self.tokenizers.get(DEFAULT_TOKENIZER) or self._ensure_default_tokenizer()
        except Exception as e:
            logger.error(f"[TokenizerRegistry] Error initiating tokenizer load for {model_name}: {str(e)}")
            return self.tokenizers.get(DEFAULT_TOKENIZER) or self._ensure_default_tokenizer()

    def _ensure_default_tokenizer(self):
        """Ensures default tokenizer exists and returns it."""
        logger.debug("[TokenizerRegistry] Ensuring default tokenizer exists")
        with self._lock:
            if DEFAULT_TOKENIZER not in self.tokenizers:
                logger.info(f"[TokenizerRegistry] Creating default tokenizer: {DEFAULT_TOKENIZER}")
                self.register_tokenizer(DEFAULT_TOKENIZER)
            return self.tokenizers[DEFAULT_TOKENIZER]

    def list_active_tokenizers(self):
        active_tokenizers = list(self.tokenizers.keys())
        return active_tokenizers
