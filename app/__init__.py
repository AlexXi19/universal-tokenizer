from flask import Flask
import logging
from app.routes import main
from app.metrics import PrometheusMiddleware


def create_app():
    app = Flask(__name__)
    
    # Register Prometheus middleware
    app.wsgi_app = PrometheusMiddleware(app.wsgi_app)
    
    # Register blueprints
    app.register_blueprint(main)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)

    app.logger.setLevel(logging.DEBUG)
    return app
