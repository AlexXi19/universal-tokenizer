# Universal Tokenizer

The **Universal Tokenizer** allows users to pass in a model name from Hugging Face or OpenAI as well as a piece of text to obtain the tokenized outputs and/or the length of the tokenized outputs.

## Features

- **Token Count API**: Supports token counting for multiple tokenizers and models.
- **Preloading Tokenizers**: Allows preloading tokenizers during server startup.
- **Active Tokenizers API**: Lists active tokenizers currently loaded in the server.
- **Environment Variable Support**: Supports `PRELOAD_TOKENIZERS` and `HF_TOKEN` for configuration.

---

## How to Use

### **Run the Server**

#### Option 1: Run from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/alexxi19/universal-tokenizer.git
   cd universal-tokenizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:

   ```bash
   python run.py
   ```

   The server will be available at `http://localhost:5000`.

---

#### Option 2: Run with Docker

You can run the Universal Tokenizer server directly from the `alexxi19/universal-tokenizer` Docker image.
w
1. **Pull the Docker image**:

   ```bash
   docker pull alexxi19/universal-tokenizer
   ```

2. **Run the Docker container**:

   You can set environment variables like `PRELOAD_TOKENIZERS` and `HF_TOKEN` for preloading models and configuring private models:

   ```bash
   docker run -e PRELOAD_TOKENIZERS="mistralai/Mistral-7B-v0.1,gpt-4o-mini" -e HF_TOKEN="<your_huggingface_token>" -p 8080:8080 alexxi19/universal-tokenizer
   ```

   The server will be available at `http://localhost:8080`.

---

## API Endpoints

### **1. Count Tokens**
**POST** `/tokenizers/count`

**Request Body**:
```json
{
  "text": "Hello world",
  "model": "bert-base-uncased"
}
```

**Response**:
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

**Response**:
```json
{
  "active_tokenizers": ["bert-base-uncased", "gpt-3.5-turbo"]
}
```

---

### Environment Variables

- `PRELOAD_TOKENIZERS`: Preload tokenizers on startup (e.g., `mistralai/Mistral-7B-v0.1,gpt-4o-mini`).
- `HF_TOKEN`: Hugging Face API token for private models.

---

### Run Tests

To run tests, use the following command:

```bash
pytest
```

--- 

