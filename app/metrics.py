from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
import time

# Request metrics
REQUEST_COUNT = Counter(
    'request_count', 
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Request latency in seconds',
    ['method', 'endpoint']
)

# Tokenizer metrics
TOKENIZER_COUNT = Counter(
    'tokenizer_count_total', 
    'Number of token counting operations',
    ['model', 'input_model']
)

TOKEN_COUNT = Counter(
    'token_count_total', 
    'Total number of tokens processed',
    ['model', 'input_model']
)

TOKENIZER_LATENCY = Histogram(
    'tokenizer_latency_seconds', 
    'Tokenizer processing time in seconds',
    ['model', 'input_model']
)

# System metrics
ACTIVE_TOKENIZERS = Gauge(
    'active_tokenizers', 
    'Number of currently loaded tokenizers'
)

TOKENIZER_INFO = Info(
    'tokenizer_service_info', 
    'Information about the tokenizer service'
)

# Set service info
TOKENIZER_INFO.info({
    'version': '1.0.0',
    'description': 'Universal Tokenizer Service'
})

class PrometheusMiddleware:
    def __init__(self, app):
        self.app = app
        
    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        
        # Skip metrics endpoint to avoid recursion
        if path == '/metrics':
            return self.app(environ, start_response)
            
        start_time = time.time()
        
        def custom_start_response(status, headers, exc_info=None):
            status_code = int(status.split(' ')[0])
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - start_time)
            return start_response(status, headers, exc_info)
            
        return self.app(environ, custom_start_response)

def get_metrics():
    """Return the latest metrics in Prometheus format"""
    return generate_latest(), CONTENT_TYPE_LATEST 
