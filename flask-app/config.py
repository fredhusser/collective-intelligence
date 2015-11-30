import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '320jWRfQhnexFPO20xBt12JwNdfz'
    SSL_DISABLE = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_RECORD_QUERIES = True
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = "fredhusser.cloud"  # os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = 'u95Bh4r3CrYp'  # os.environ.get('MAIL_PASSWORD')
    FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
    FLASKY_MAIL_SENDER = 'Flasky Admin <fredhusser.cloud@gmail.com>'
    FLASKY_ADMIN = 'fredhusser'  # os.environ.get('FLASKY_ADMIN')
    FLASKY_POSTS_PER_PAGE = 20
    FLASKY_FOLLOWERS_PER_PAGE = 50
    FLASKY_COMMENTS_PER_PAGE = 30
    FLASKY_SLOW_DB_QUERY_TIME = 0.5

    @staticmethod
    def init_app(app):
        pass


class MongoConfig:
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    MONGODB_DATABASE = 'collective_intelligence'
    MONGODB_COLLECTION = 'som_classifier'
    def __init__(self):
        from pymongo import MongoClient
        connection = MongoClient(self.MONGODB_HOST, self.MONGODB_PORT)
        self.collection = connection[self.MONGODB_DATABASE][self.MONGODB_COLLECTION]


class DevelopmentConfig(Config):
    DEBUG = True

    MONGODB_SERVER_HOST = os.environ.get('MONGO_PORT_27017_TCP_ADDR',"localhost")
    MONGODB_SERVER_PORT = os.environ.get('MONGO_PORT_27017_TCP_PORT',27017)
    MONGODB_DATABASE = 'collective_intelligence'
    MONGODB_COLLECTION = 'som_classifier'

    POSTGRES_SERVER_HOST = os.environ.get('POSTGRES_PORT_5432_TCP_ADDR',"localhost")
    POSTGRES_SERVER_PORT = os.environ.get('POSTGRES_PORT_5432_TCP_PORT',5432)
    POSTGRES_SERVER_DB = "postgres"
    POSTGRES_SERVER_PASSWORD = ""
    SQLALCHEMY_DATABASE_URI = "postgresql://"+POSTGRES_SERVER_HOST+":"+str(POSTGRES_SERVER_PORT)+\
                              "/"+POSTGRES_SERVER_DB


config = {
    'development': DevelopmentConfig,
    'default': DevelopmentConfig
}
