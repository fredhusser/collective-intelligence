"""API interface to fetch nodes data.
"""
__author__ = 'husser'

from flask import jsonify, request, g, abort, url_for, current_app
from .. import db
from ..models import Node, Topic, Cluster
from . import api

@api.route('/nodes/')

@api.route('/context/<int:context_id>/nodes/<int:id>')
def get_node(id, context_id):
    # Todo: jsonify get_node
    return jsonify({})


@api.route('/node/<int:id>/posts')
def get_node_posts(id):
    # Todo: get the JSON of all posts corresponding to a node
    return jsonify({})


@api.route('/cluster/<int:id>')
def get_cluster(id):
    # Todo get the JSON for a cluster
    return jsonify({})


@api.route('/cluster/<int:id>/nodes')
def get_cluster_nodes(id):
    # Todo get the JSON for the nodes given cluster ID
    return jsonify({})


@api.route('/topic/<int:id>')
def get_topic(id):
    # Todo get topic by id
    return jsonify({})


@api.route('/topic/<int:id>/nodes')
def get_topic(id):
    # Todo get nodes by topic id
    return jsonify({})


@api.route('/topic/<int:id>/posts')
def get_topic(id):
    # Todo get posts by topic id
    return jsonify({})

