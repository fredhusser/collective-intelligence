__author__ = 'husser'
from flask import Blueprint

viz = Blueprint('viz', __name__)

from . import views, dataviz
