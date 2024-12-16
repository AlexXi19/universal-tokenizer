from app.services.huggingface_tokenizer import HuggingFaceTokenizer
from app.services.openai_tokenizer import OpenAITokenizer
import tiktoken
from app.services.logger import logger

DEFAULT_TOKENIZER = 'o200k_base'


class TokenizerRegistry:
    def __init__(self, preload_tokenizers=None):
        self.tokenizers = {}
        self._tokenizer_type_cache = {}
        if preload_tokenizers:
            self._preload_tokenizers(preload_tokenizers)

    def _preload_tokenizers(self, preload_tokenizers):
        for model_name in preload_tokenizers:
            self.register_tokenizer(model_name)

    def get_tokenizer_type(self, model_name: str):
        if model_name in self._tokenizer_type_cache:
            return self._tokenizer_type_cache[model_name]

        try:
            try:
                tiktoken.encoding_for_model(model_name)
                tokenizer_type = "openai"
            except KeyError:
                tiktoken.get_encoding(model_name)
                tokenizer_type = "openai"
        except (KeyError, ValueError):
            tokenizer_type = "huggingface"

        self._tokenizer_type_cache[model_name] = tokenizer_type
        return tokenizer_type

    def register_tokenizer(self, model_name: str):
        tokenizer_type = self.get_tokenizer_type(model_name)
        if tokenizer_type == "huggingface":
            self.tokenizers[model_name] = HuggingFaceTokenizer(model_name)
        elif tokenizer_type == "openai":
            self.tokenizers[model_name] = OpenAITokenizer(model_name)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        logger.info(f"Tokenizer registered: {model_name}")

    def get_tokenizer(self, model_name: str):
        if model_name not in self.tokenizers:
            try:
                self.register_tokenizer(model_name)
            except (KeyError, ValueError, OSError) as e:
                if DEFAULT_TOKENIZER not in self.tokenizers:
                    self.register_tokenizer(DEFAULT_TOKENIZER)
                return self.tokenizers[DEFAULT_TOKENIZER]
        return self.tokenizers[model_name]

    def list_active_tokenizers(self):
        return list(self.tokenizers.keys())
