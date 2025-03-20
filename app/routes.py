from flask import Blueprint, request, jsonify, Response
from app.services.tokenizer_registry import TokenizerRegistry
from app.services.logger import logger
import os
import time
from app.metrics import (
    ACTIVE_TOKENIZERS, track_tokens
)

preload_tokenizers = [t.strip() for t in os.getenv(
    "PRELOAD_TOKENIZERS", "").split(",") if t.strip()]

main = Blueprint('main', __name__)
registry = TokenizerRegistry(preload_tokenizers=preload_tokenizers)

# Update active tokenizers gauge (will be done after metrics initialization)


@main.route('/')
def home():
    return "Universal Tokenizer™️"


@main.route('/health')
def health():
    return "ok"


# Metrics endpoint is automatically added by prometheus-flask-exporter


@main.route('/tokenizers/count', methods=['POST'])
def count_tokens():
    try:
        data = request.json
        text = data.get("text", "")
        model_name = data.get("model", "")

        if not model_name:
            raise ValueError("Field 'model' is required")

        # Measure tokenization time
        start_time = time.time()
        tokenizer = registry.get_tokenizer(model_name)
        if not text:
            result = {"token_count": 0,
                      "model": tokenizer.model_name, "tokenizer": "openai"}
        else:
            result = tokenizer.count_tokens(text)

        # Calculate duration
        duration = time.time() - start_time
        
        # Record all metrics with one call
        track_tokens(
            tokenizer_model=tokenizer.model_name,
            input_model=model_name,
            token_count=result.get("token_count", 0),
            duration=duration
        )

        logger.info(
                f"Token count request for {model_name} with {result.get('token_count', 0)} tokens completed in {duration:.2f}s")

        return jsonify(result)

    except ValueError as e:
        logger.warning(f"Validation error in count_tokens: {str(e)} - Request data: {data}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(
            f"Error processing count_tokens request: Request data: {data}")
        return jsonify({"error": "Internal server error: " + str(e)}), 500


@main.route('/tokenizers/list', methods=['GET'])
def list_active_tokenizers():
    active_models = registry.list_active_tokenizers()
    # Update active tokenizers gauge
    ACTIVE_TOKENIZERS.set(len(active_models))
    return jsonify({"active_tokenizers": active_models})
