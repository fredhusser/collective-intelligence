FROM ubuntu:14.04
MAINTAINER Frederic Husser

COPY ./ /opt/scraper
WORKDIR /opt/scraper

RUN apt-get update
RUN apt-get install -y build-essential python-dev python-pip python-scrapy
RUN pip install -r /opt/scraper/requirements.txt

EXPOSE 6080

ENTRYPOINT ["scrapy", "crawl", "lemonde"]
