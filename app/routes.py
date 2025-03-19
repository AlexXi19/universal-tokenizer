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

        # Record latency
        TOKENIZER_LATENCY.labels(model=tokenizer.model_name, input_model=model_name).observe(
            time.time() - start_time)

        # Increment counter for tokenizer usage
        TOKENIZER_COUNT.labels(model=tokenizer.model_name,
                               input_model=model_name).inc()

        # Record token count
        TOKEN_COUNT.labels(model=tokenizer.model_name, input_model=model_name).inc(
            result.get("token_count", 0))

        logger.info(
                f"Token count request for {model_name} with {result.get('token_count', 0)} tokens completed in {time.time() - start_time:.2f}s")

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
