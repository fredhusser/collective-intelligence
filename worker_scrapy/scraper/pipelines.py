from scrapy.exceptions import DropItem
from scrapy.conf import settings
from scrapy import log
import bleach
import datetime
import pymongo
import re

tags = ['h2', 'p', 'em', 'strong']

class WebArticlesPipeline(object):
    """A pipeline for storing scraped items in the database"""
    MONGODB_COLLECTION = "lemonde"

    def __init__(self, collection_name='lemonde'):
        connection = pymongo.MongoClient(
            settings['MONGODB_SERVER_HOST'],
            settings['MONGODB_SERVER_PORT']
        )
        db = connection[settings["MONGODB_DB"]]
        self.collection = db[collection_name]

    def process_item(self, item, spider):
        """This method is called for every item pipeline component.
        """
        item["timestamp"]=item["timestamp"].split("+")[0]
        item["body"]=bleach.clean(item["body"], tags, strip=True)
        valid = True

        for data in item:
            if not data:
                valid = False
                raise DropItem("Missing {0}!".format(data))

        if valid:
            self.collection.insert(dict(item))
            log.msg("Question added to MongoDB database",
                    level = log.DEBUG, spider=spider)
        return item