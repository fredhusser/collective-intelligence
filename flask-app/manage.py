#!/usr/bin/env python
import os
from app import create_app, db
from flask.ext.script import Manager, Server

if os.path.exists('.env'):
    print('Importing environment from .env...')
    for line in open('.env'):
        var = line.strip().split('=', 1)
        if len(var) == 2:
            os.environ.setdefault(var[0], var[1])

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
manager = Manager(app)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
