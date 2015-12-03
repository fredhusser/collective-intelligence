from flask import render_template
from bson.objectid import ObjectId
from app.models import Post
from . import posts
from app import scrapy


@posts.route("/", methods=["GET", "POST"])
def list_posts():
    posts = scrapy.lemonde.find()
    return render_template('posts/posts.html', posts=posts)


@posts.route("/post/<post_id>/", methods=['GET', 'POSTS'])
def get_post(post_id):
    post = scrapy.lemonde.find_one({"_id":ObjectId(post_id)})
    print post_id, post
    return render_template('posts/post.html', post=post)


