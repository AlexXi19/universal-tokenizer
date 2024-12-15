from transformers import AutoTokenizer
from app.services.base_tokenizer import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> dict:
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return {
            "token_count": len(input_ids),
            "model": self.model_name,
            "tokenizer": "huggingface"
        }
