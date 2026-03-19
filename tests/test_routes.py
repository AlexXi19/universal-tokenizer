import json
import os
import pytest


def _hf_available():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-32B-Instruct")
        return True
    except Exception:
        return False


requires_hf = pytest.mark.skipif(
    not _hf_available(),
    reason="HuggingFace model not available (set HF_TOKEN or check network)",
)


# ── OpenAI ───────────────────────────────────────────────────────────────────


def test_count_tokens_openai(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world", "model": "gpt-4o"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["tokenizer"] == "openai"


def test_count_tokens_openai_tokenizer(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world", "model": "o200k_base"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["model"] == "o200k_base"
    assert data["tokenizer"] == "openai"


# ── HuggingFace (requires network + optional HF_TOKEN) ──────────────────────


@requires_hf
def test_count_tokens_huggingface(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps(
            {"text": "Hello world", "model": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"}
        ),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["tokenizer"] in ("huggingface", "openai")


# ── Gemini ───────────────────────────────────────────────────────────────────


def test_count_tokens_gemini(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world", "model": "gemini-2.0-flash"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["tokenizer"] in ("gemini", "openai")


def test_count_tokens_gemini_25(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world", "model": "gemini-2.5-flash"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["tokenizer"] in ("gemini", "openai")


def test_count_tokens_gemini_empty_text(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "", "model": "gemini-2.0-flash"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] == 0


def test_gemini_tokenizer_unit():
    from app.services.gemini_tokenizer import GeminiTokenizer

    tokenizer = GeminiTokenizer("gemini-2.0-flash")
    result = tokenizer.count_tokens("Hello world")
    assert result["token_count"] > 0
    assert result["model"] == "gemini-2.0-flash"
    assert result["tokenizer"] == "gemini"


def test_gemini_tokenizer_chinese():
    from app.services.gemini_tokenizer import GeminiTokenizer

    tokenizer = GeminiTokenizer("gemini-2.0-flash")
    result = tokenizer.count_tokens("你好世界")
    assert result["token_count"] > 0
    assert result["tokenizer"] == "gemini"


def test_gemini_tokenizer_long_text():
    from app.services.gemini_tokenizer import GeminiTokenizer

    tokenizer = GeminiTokenizer("gemini-2.0-flash")
    text = "This is a test sentence. " * 100
    result = tokenizer.count_tokens(text)
    assert result["token_count"] > 100
    assert result["tokenizer"] == "gemini"


# ── Fallback & validation ───────────────────────────────────────────────────


def test_count_tokens_nonexistent_tokenizer(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world", "model": "unknown"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["model"] == "o200k_base"
    assert data["tokenizer"] == "openai"


def test_missing_fields(client):
    response = client.post(
        "/tokenizers/count",
        data=json.dumps({"text": "Hello world"}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_list_active_tokenizers(client):
    response = client.get("/tokenizers/list")
    assert response.status_code == 200
    data = response.get_json()
    assert "active_tokenizers" in data
    assert "o200k_base" in data["active_tokenizers"]
