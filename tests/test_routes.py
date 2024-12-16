import json


def test_count_tokens_huggingface(client):
    response = client.post(
        '/tokenizers/count',
        data=json.dumps({
            "text": "Hello world",
            "model": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] == 2
    assert data["model"] == "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    assert data["tokenizer"] == "huggingface"


def test_count_tokens_openai(client):
    response = client.post(
        '/tokenizers/count',
        data=json.dumps({
            "text": "Hello world",
            "model": "gpt-4o"
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["model"] == "gpt-4o"
    assert data["tokenizer"] == "openai"


def test_count_tokens_openai_tokenizer(client):
    response = client.post(
        '/tokenizers/count',
        data=json.dumps({
            "text": "Hello world",
            "model": "o200k_base"
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["model"] == "o200k_base"
    assert data["tokenizer"] == "openai"

def test_count_tokens_nonexistent_tokenizer(client):
    response = client.post(
        '/tokenizers/count',
        data=json.dumps({
            "text": "Hello world",
            "model": "unknown"
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["token_count"] > 0
    assert data["model"] == "o200k_base"
    assert data["tokenizer"] == "openai"


def test_missing_fields(client):
    response = client.post(
        '/tokenizers/count',
        data=json.dumps({"text": "Hello world"}),
        content_type='application/json'
    )
    assert response.status_code == 400


def test_list_active_tokenizers(client):
    response = client.get('/tokenizers/list')
    assert response.status_code == 200
    data = response.get_json()
    assert "active_tokenizers" in data
    assert "LGAI-EXAONE/EXAONE-3.5-32B-Instruct" in data["active_tokenizers"]
    assert "Sao10K/14B-Qwen2.5-Kunou-v1" in data["active_tokenizers"]
