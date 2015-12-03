from flask import Flask
from flask.ext.bootstrap import Bootstrap
from flask.ext.pagedown import PageDown
from flask.ext.mongoengine import MongoEngine
from flask_wtf.csrf import CsrfProtect
from config import config
import pymongo
import os

bootstrap = Bootstrap()
pagedown = PageDown()
db = MongoEngine()
csrf = CsrfProtect()

conn = pymongo.MongoClient(os.environ.get("MONGODB_SERVER_HOST"),
                          os.environ.get("MONGODB_SERVER_PORT"))
scrapy = conn.scrapy


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    bootstrap.init_app(app)
    pagedown.init_app(app)
    db.init_app(app)
    csrf.init_app(app)

    print db

    from .main import posts as posts_blueprint
    app.register_blueprint(posts_blueprint)

    return app
