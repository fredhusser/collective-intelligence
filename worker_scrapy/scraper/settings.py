# Scrapy settings for dirbot project
import os

#BOT_NAME = 'lemonde'
SPIDER_MODULES = ['scraper.spiders']
NEWSPIDER_MODULE = 'scraper.spiders'
DEFAULT_ITEM_CLASS = 'scraper.items.LeMondeArt'

ITEM_PIPELINES = {# Pipelines are used for filtering data
                  'scraper.pipelines.WebArticlesPipeline': 1
                  }


MONGODB_SERVER_HOST = os.environ.get('MONGO_PORT_27017_TCP_ADDR', "localhost")
MONGODB_SERVER_PORT = os.environ.get('MONGO_PORT_27017_TCP_PORT',27017)
MONGODB_DB = "scrapy"