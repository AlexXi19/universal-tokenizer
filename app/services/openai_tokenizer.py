import tiktoken
from app.services.base_tokenizer import BaseTokenizer


class OpenAITokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            try:
                self.encoder = tiktoken.get_encoding(model_name)
            except KeyError:
                raise ValueError(
                    f"Invalid model or tokenizer name: {model_name}")

    def count_tokens(self, text: str) -> dict:
        tokens = self.encoder.encode(text)
        return {
            "token_count": len(tokens),
            "model": self.model_name,
            "tokenizer": "openai"
        }
