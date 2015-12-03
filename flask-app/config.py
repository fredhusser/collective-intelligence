import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SSL_DISABLE = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_RECORD_QUERIES = True
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
    FLASKY_MAIL_SENDER = 'Flasky Admin <fredhusser.cloud@gmail.com>'
    FLASKY_ADMIN = os.environ.get('FLASKY_ADMIN')
    FLASKY_POSTS_PER_PAGE = 20
    FLASKY_FOLLOWERS_PER_PAGE = 50
    FLASKY_COMMENTS_PER_PAGE = 30
    FLASKY_SLOW_DB_QUERY_TIME = 0.5

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    MONGODB_HOST = os.environ.get('MONGO_PORT_27017_TCP_ADDR', "localhost")
    MONGODB_PORT = int(os.environ.get('MONGO_PORT_27017_TCP_PORT', 27017))
    MONGODB_DB = 'scrapy'
    MONGODB_COLLECTION = 'som_classifier'
    MONGODB_USERNAME = ""
    MONGODB_PASSWORD = ""


    POSTGRES_SERVER_HOST = os.environ.get('POSTGRES_PORT_5432_TCP_ADDR', 'localhost')
    POSTGRES_SERVER_PORT = int(os.environ.get('POSTGRES_PORT_5432_TCP_PORT', 5432))
    POSTGRES_DB = os.environ.get('POSTGRES_DBNAME', 'postgres')
    POSTGRES_USER = os.environ.get('POSTGRES_USER', 'postgres')
    POSTGRES_SERVER_PASSWORD = ""
    SQLALCHEMY_DATABASE_URI = "postgresql://" + POSTGRES_USER + "@" + POSTGRES_SERVER_HOST + ":" \
                              + str(POSTGRES_SERVER_PORT) + "/" + POSTGRES_DB


config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}
