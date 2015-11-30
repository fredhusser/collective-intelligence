from scrapy.item import Item, Field


class LeMondeArt(Item):
    """Define the model of the articles extracted from the website.
    """
    title = Field()
    timestamp = Field()
    body = Field()
    
    
class LeMondeCat(Item):
    """Define the categories appearing in the main page from where
    data is to be scraped. Categories are found from hyperlinks in the menu
    and from the second categories.
    """
    Name = Field()
    Url = Field()
    Level = Field()