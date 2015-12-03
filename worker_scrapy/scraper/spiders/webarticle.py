#!/usr/bin/env python
# encoding=utf-8

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors import LinkExtractor
from scrapy.contrib.loader.processor import Join, MapCompose
from scrapy.contrib.loader import XPathItemLoader

from scrapy.selector import Selector

from ..items import LeMondeArt, LeMondeCat



class LeMondeSpider(CrawlSpider):
    name = "lemonde"
    allowed_domains = ["lemonde.fr"]
    start_urls = [
        "http://www.lemonde.fr/",
    ]
    
    article_item_fields = {
        'title': './/article/h1/text()',
        #'Author': './/article/p[@class="bloc_signature"]/span[@class="signature_article"]/span[@itemprop="author"]/a.text()',
        #'Publisher': './/article/p[@class="bloc_signature"]/span[@id="publisher"]/text()',
        'timestamp': './/article/p[@class="bloc_signature"]/time[@itemprop="datePublished"]/@datetime',
        'body': './/article/div[@id="articleBody"]/*',
    }
    
    rules = (
             # Rule(LinkExtractor(allow=[r'w+']), follow=True), 
             # Extract links matching to the article link
             Rule(LinkExtractor(allow=(r"article/\d{4}/\d{2}/\d{2}/.+")), callback = "parse_article", follow = True),
            )

    def parse_article(self, response):
        """
        The lines below is a spider contract. For more info see:
        http://doc.scrapy.org/en/latest/topics/contracts.html

        @url http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/
        @scrapes name
        """
        
        selector = Selector(response)
        loader = XPathItemLoader(LeMondeArt(), selector=selector)
        
        self.log('\n\nA response from %s just arrived!' % response.url)
        
        # define processors
        text_input_processor = MapCompose(unicode.strip)
        loader.default_output_processor = Join()
        
        # Populate the LeMonde Item with the item loader
        for field, xpath in self.article_item_fields.iteritems():
            try:
                loader.add_xpath(field, xpath, text_input_processor)
            except ValueError:
                self.log("XPath %s not found at url %s" % (xpath, response.url))
            
        #loader.add_value("Url",response.url)
        

        yield loader.load_item()
