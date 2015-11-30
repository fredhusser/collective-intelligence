__author__ = 'husser'
from flask import render_template, redirect, url_for, flash
from bokeh.embed import components
import pandas as pd
from . import viz, dataviz
from .. import db, mongo
from ..models import Post



@viz.route('/')
def som_graph():
    # Add some data to the blueprint
    with pd.HDFStore('store.h5') as hdf_store:
        nodes_data = pd.read_hdf(hdf_store, "data/nodes_data")
        context = pd.read_hdf(hdf_store, 'context').ix[-1].to_dict()
        n_clusters = context['n_clusters']
        k_shape = (context['x_shape'], context['y_shape'])
        source, x_range, y_range = dataviz.get_grid(nodes_data, {"n_clusters": n_clusters,
                                                                 "kshape": k_shape})
        p = dataviz.render_grid(source, x_range, y_range)
    script, div = components(p)
    return render_template('dataviz/som_graph.html', script=script, div=div)


@viz.route('/node/<int:node_id>', methods=['GET', 'POST'])
def som_node(node_id):
    data = mongo.find_one({'nodes.id':node_id},{'nodes':{"$elemMatch": {"id":node_id}}}).get("nodes")
    if data!= []:
        post_ids = data[0].get("articles")
        posts = db.session.query(Post).filter(Post.id.in_(post_ids)).all()
        return render_template('dataviz/som_node.html',
                           node=data[0],
                           posts=posts)
