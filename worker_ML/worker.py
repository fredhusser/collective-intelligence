__author__ = 'husser'
## To do : use the Flask API to fetch data

import sys
from optparse import OptionParser
from textmining.workflows import SemanticAnalysisWorkflow
from pymongo import MongoClient
from config import MONGODB_SERVER_HOST, MONGODB_SERVER_PORT

mongoclient = MongoClient(MONGODB_SERVER_HOST, MONGODB_SERVER_PORT)

op = OptionParser()
op.add_option("--use_nmf",
              action="store_true", dest="nmf", default=False,
              help="Use the nmf dimensionality reduction")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

if opts.nmf:
    reducer = "nmf"
else:
    reducer = "lsa"
print("Using reducer %s"%reducer)

if __name__ == '__main__':
    corpus = SemanticAnalysisWorkflow(mongoclient, "scrapy", "lemonde")
    corpus.main()
