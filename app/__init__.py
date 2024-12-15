from flask import Flask
import logging


def create_app():
    app = Flask(__name__)
    from app.routes import main
    app.register_blueprint(main)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)

    app.logger.setLevel(logging.DEBUG)
    return app
