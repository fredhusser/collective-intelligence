__author__ = 'husser'

from flask.ext.wtf import Form
from wtforms import StringField, TextAreaField, BooleanField, SelectField,\
    SubmitField
from wtforms.validators import Required, Length, Email, Regexp
from wtforms import ValidationError
from flask.ext.pagedown.fields import PageDownField
from app.models import Role, User


class RunTextMining(Form):
    submit = SubmitField('Start')
