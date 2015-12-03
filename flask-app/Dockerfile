FROM ubuntu:12.04
MAINTAINER Chris Brantley

COPY ./ /opt/flask-app/

RUN apt-get update
RUN apt-get install -y build-essential python-pip
RUN pip install -r /opt/flask-app/requirements.txt

WORKDIR /opt/flask-app

EXPOSE 80
