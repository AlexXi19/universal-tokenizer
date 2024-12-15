# AutuTokenizer Server

The AutoTokenizer Server allows users to pass in a model name from Hugging Face OpenAI as well as a piece of text to obtain the tokenized outputs and/or the length of the tokenized outputs.


## Features

- **Token Count API**: Supports token counting for multiple tokenizers and models.
- **Preloading Tokenizers**: Allows preloading tokenizers during server startup.
- **Active Tokenizers API**: Lists active tokenizers currently loaded in the server.
- **Environment Variable Support**: Supports `PRELOAD_TOKENIZERS` and `HF_TOKEN` for configuration.

---

## How to Use

### **Run the Server**
```bash
python run.py
```

The server will run at `http://localhost:5000`.

---

## API Endpoints

### **1. Count Tokens**
**POST** `/tokenizers/count`
```json
{
  "text": "Hello world",
  "model": "bert-base-uncased"
}
```

**Response**
```json
{
  "token_count": 3,
  "model": "bert-base-uncased",
  "tokenizer": "huggingface"
}
```

---

### **2. List Active Tokenizers**
**GET** `/tokenizers/list/active`

**Response**
```json
{
  "active_tokenizers": ["bert-base-uncased", "gpt-3.5-turbo"]
}
```

---

### Environment Variables

- `PRELOAD_TOKENIZERS`: Preload tokenizers on startup (e.g., `huggingface:bert-base-uncased,openai:gpt-3.5-turbo`).
- `HF_TOKEN`: Hugging Face API token for private models.

---

### Run Tests
```bash
pytest
```

---

