from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge, multiprocess
from flask import request
import os

# Set up multiprocess directory if not already set
if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = '/tmp/prometheus_multiproc'
    # Ensure directory exists
    os.makedirs('/tmp/prometheus_multiproc', exist_ok=True)

# Initialize these as None - they'll be set in init_metrics
metrics = None
TOKENIZER_COUNT = None
TOKEN_COUNT = None
TOKENIZER_LATENCY = None
ACTIVE_TOKENIZERS = None

def init_metrics(app):
    """Initialize metrics with Flask app"""
    # Initialize with the Flask app
    global metrics, TOKENIZER_COUNT, TOKEN_COUNT, TOKENIZER_LATENCY, ACTIVE_TOKENIZERS
    
    # Create metrics instance with the app - use Gunicorn multiprocess version
    metrics = GunicornInternalPrometheusMetrics(app)
    
    # Add app info
    metrics.info('app_info', 'Application info', version='1.0.0')
    
    # Create metrics using direct prometheus-client classes
    # This gives us access to the correct methods like .labels().inc()
    TOKENIZER_COUNT = Counter(
        'tokenizer_count_total', 
        'Number of token counting operations',
        ['model', 'input_model'],
        registry=metrics.registry
    )
    
    TOKEN_COUNT = Counter(
        'token_count_total', 
        'Total number of tokens processed',
        ['model', 'input_model'],
        registry=metrics.registry
    )
    
    TOKENIZER_LATENCY = Histogram(
        'tokenizer_latency_seconds', 
        'Tokenizer processing time in seconds',
        ['model', 'input_model'],
        registry=metrics.registry
    )
    
    # System metrics - use direct prometheus-client for the gauge
    ACTIVE_TOKENIZERS = Gauge(
        'active_tokenizers', 
        'Number of currently loaded tokenizers',
        registry=metrics.registry
    )
    
    # Service info metric - avoid duplicate description
    metrics.info(
        'tokenizer_service_info', 
        'Universal Tokenizer Service',
        version='1.0.0'
    )
    
    return metrics

# Helper functions for working with the metrics
def track_tokens(tokenizer_model, input_model, token_count, duration):
    """Record all tokenizer-related metrics in one call"""
    # Use direct labels with the prometheus-client metrics
    labels = {
        'model': tokenizer_model, 
        'input_model': input_model
    }
    
    # Increment the counter for tokenizer usage
    TOKENIZER_COUNT.labels(
        model=tokenizer_model,
        input_model=input_model
    ).inc()
    
    # Record the token count
    TOKEN_COUNT.labels(
        model=tokenizer_model, 
        input_model=input_model
    ).inc(token_count)
    
    # Record the latency
    TOKENIZER_LATENCY.labels(
        model=tokenizer_model, 
        input_model=input_model
    ).observe(duration)

def get_metrics():
    """For compatibility with existing code - not needed with flask-exporter"""
    return metrics.generate_latest(), metrics.content_type 
