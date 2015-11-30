__author__ = 'husser'
## To do : use the Flask API to fetch data

from optparse import OptionParser
import sys

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from textmining.workflows import MainWorkflow

import pymongo

engine = create_engine("sqlite:///data-dev.db")
DBSession = sessionmaker()
DBSession.configure(bind=engine)
session = DBSession()

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
    corpus = MainWorkflow().read_sql(session, Post, size_limit=2000)
    corpus.run_main()
    corpus.aggregate_map_data(n_topics=4, n_features=5)
    corpus.store_in_mongo()