import pytest
from app import create_app
import os

os.environ["PRELOAD_TOKENIZERS"] = "Sao10K/14B-Qwen2.5-Kunou-v1,o200k_base"


@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    return app


@pytest.fixture
def client(app):
    return app.test_client()
