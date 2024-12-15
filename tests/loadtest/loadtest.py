from locust import HttpUser, task, between


class TokenizerLoadTest(HttpUser):
    # Time between tasks
    wait_time = between(1, 2)

    @task
    def count_tokens(self):
        headers = {'Content-Type': 'application/json'}
        data = {
            "text": "OpenAIs large language models process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. Learn more. OpenAIs large language models process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. Learn more. OpenAIs large language models process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. Learn more.",
            "model": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
        }
        self.client.post("/tokenizers/count", json=data, headers=headers)
