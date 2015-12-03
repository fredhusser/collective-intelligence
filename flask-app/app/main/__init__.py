from flask import Blueprint

posts = Blueprint('posts', __name__, template_folder='app/templates')

from . import views, auth