import os

MONGODB_SERVER_HOST = os.environ.get('MONGO_PORT_27017_TCP_ADDR',"localhost")
MONGODB_SERVER_PORT = os.environ.get('MONGO_PORT_27017_TCP_PORT',27017)
MONGODB_DATABASE = 'collective_intelligence'
MONGODB_COLLECTION = 'som_classifier'


if __name__ == "__main__":
    print "Host:",MONGODB_SERVER_HOST
    print "Port:",MONGODB_SERVER_PORT