from flask import Blueprint, request, jsonify
from app.services.tokenizer_registry import TokenizerRegistry
from app.services.logger import logger
import os

preload_tokenizers = [t.strip() for t in os.getenv(
    "PRELOAD_TOKENIZERS", "").split(",") if t.strip()]

main = Blueprint('main', __name__)
registry = TokenizerRegistry(preload_tokenizers=preload_tokenizers)


@main.route('/')
def home():
    return "Universal Tokenizer™️"


@main.route('/health')
def health():
    return "ok"


@main.route('/tokenizers/count', methods=['POST'])
def count_tokens():
    try:
        data = request.json
        text = data.get("text", "")
        model_name = data.get("model", "")

        if not text or not model_name:
            raise ValueError("Fields 'text' and 'model' are required")

        tokenizer = registry.get_tokenizer(model_name)
        result = tokenizer.count_tokens(text)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": "Internal server error: " + str(e)}), 500


@main.route('/tokenizers/list', methods=['GET'])
def list_active_tokenizers():
    active_models = registry.list_active_tokenizers()
    return jsonify({"active_tokenizers": active_models})
