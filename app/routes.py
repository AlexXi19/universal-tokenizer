from flask import Blueprint, request, jsonify, Response
from app.services.tokenizer_registry import TokenizerRegistry
from app.services.logger import logger
import os
import time
from app.metrics import (
    TOKENIZER_COUNT, TOKEN_COUNT, TOKENIZER_LATENCY, 
    ACTIVE_TOKENIZERS, get_metrics
)

preload_tokenizers = [t.strip() for t in os.getenv(
    "PRELOAD_TOKENIZERS", "").split(",") if t.strip()]

main = Blueprint('main', __name__)
registry = TokenizerRegistry(preload_tokenizers=preload_tokenizers)

# Update active tokenizers gauge
ACTIVE_TOKENIZERS.set(len(registry.list_active_tokenizers()))

@main.route('/')
def home():
    return "Universal Tokenizer™️"


@main.route('/health')
def health():
    return "ok"


@main.route('/metrics')
def metrics():
    # Update active tokenizers count before returning metrics
    ACTIVE_TOKENIZERS.set(len(registry.list_active_tokenizers()))
    return Response(get_metrics()[0], mimetype=get_metrics()[1])


@main.route('/tokenizers/count', methods=['POST'])
def count_tokens():
    try:
        data = request.json
        text = data.get("text", "")
        model_name = data.get("model", "")

        if not text or not model_name:
            raise ValueError("Fields 'text' and 'model' are required")

        # Increment counter for tokenizer usage
        TOKENIZER_COUNT.labels(model=model_name).inc()
        
        # Measure tokenization time
        start_time = time.time()
        tokenizer = registry.get_tokenizer(model_name)
        result = tokenizer.count_tokens(text)
        
        # Record latency
        TOKENIZER_LATENCY.labels(model=model_name).observe(time.time() - start_time)
        
        # Record token count
        TOKEN_COUNT.labels(model=model_name).inc(result.get("token_count", 0))
        
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": "Internal server error: " + str(e)}), 500


@main.route('/tokenizers/list', methods=['GET'])
def list_active_tokenizers():
    active_models = registry.list_active_tokenizers()
    # Update active tokenizers gauge
    ACTIVE_TOKENIZERS.set(len(active_models))
    return jsonify({"active_tokenizers": active_models})
