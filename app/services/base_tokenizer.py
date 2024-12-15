from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    @abstractmethod
    def count_tokens(self, model_name: str, text: str) -> dict:
        pass

