#!/usr/bin/env bash

#update package lists
sudo apt-get update

file="Anaconda-2.1.0-Linux-x86_64.sh"
if [ -f "$file" ]
then
	echo "$file found."
else
	echo "$file not found."
	sudo wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh
fi

sudo bash Anaconda-2.1.0-Linux-x86_64.sh -b
export PATH="/home/vagrant/anaconda/bin:$PATH"
sudo conda update -y conda
sudo conda install -y pymongo

# Create a virtual environment for scrapy
cd /vagrant/worker_scrapy
sudo conda create -n scrapyenv -y scrapy pymongo

# Create a virtual environment for development of the flask application
cd /vagrant/flask-app
sudo conda create -n flaskenv -y libconda
source activate flaskenv
pip install -y -r requirements.txt
source deactivate

#install the pip python package manager
#sudo apt-get install -y python-pip vim git-core screen unzip libyaml-dev
#sudo pip install distribute
#sudo pip install nltk
#sudo python -m nltk.downloader all

###################################################
## Python Port of Stanford NLP libraries         ##
## https://bitbucket.org/torotoki/corenlp-python ##
###################################################
#install prerequisites
# sudo pip install pexpect unidecode jsonrpclib
#clone the repository and download datafiles
# git clone https://bitbucket.org/torotoki/corenlp-python.git
# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2013-06-20.zip
# unzip stanford-corenlp-full-2013-06-20.zip