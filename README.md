# Collective Intelligence project
A multi service environment for programming collective intelligence application in python. The mission of this project is to mine news and opinions articles in the web for providing:
- analytics on articles (classification, clustering)
- articles recommendation based on the content and social networks;
- new browsing experience for information and news.

## Project Structure 

This project is based on Vagrant for providing a development environment, provisioned with the following:
- Virtual machine: running Ubuntu;
- Python distribution: Anaconda from Continuum Analytics. Anaconda was chosen for it is well adapted to data analysis projects, with all necessary dependancies (Numpy, Scipy, Pandas, Scikit-Learn, Ipython). We use then virtual environments for the specific needs of each analytics service.
- Docker for setting up services: MongoDB, Postgres SQL, a Flask powered web application and Nginx. Docker is used for its ability to provide specific services in a well contained environment (containers) and we rely on images located on the DockerHub platform. Additional specific services such as ElasticSearch or Google TensorFlow can be provided through Docker containers.

## Project functionalities

This project follows the full data analysis process:
- data collection (using Scrapy);
- data analysis workers for machine learning tasks : text mining, classification and clustering;
- data bulk storage in MongoDB database;
- serving results by a Flask application with a Postgres SQL database;
- data visualization with D3.js


For development environment settings look: [http://devbandit.com/2015/05/29/vagrant-and-docker.html](http://devbandit.com/2015/05/29/vagrant-and-docker.html)

## Install

First, install the [vagrant-docker-compose](https://github.com/leighmcculloch/vagrant-docker-compose) plugin (as root).

```bash
vagrant plugin install vagrant-docker-compose
```

Then run `vagrant up`.

'''bash

''' 
