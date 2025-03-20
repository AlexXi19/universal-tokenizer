from flask import Flask
import logging
from app.metrics import init_metrics

def create_app():
    app = Flask(__name__)
    
    # Initialize Prometheus metrics first
    metrics = init_metrics(app)
    
    # Import routes after metrics are initialized
    from app.routes import main
    
    # Register blueprints
    app.register_blueprint(main)

    # Set up logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    
    # Initialize active tokenizers after metrics are set up
    from app.metrics import ACTIVE_TOKENIZERS
    from app.routes import registry
    ACTIVE_TOKENIZERS.set(len(registry.list_active_tokenizers()))
    
    return app
