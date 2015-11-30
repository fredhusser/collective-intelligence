from datetime import datetime
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from markdown import markdown
import bleach
from flask import current_app, request, url_for, jsonify
from flask.ext.login import UserMixin, AnonymousUserMixin
from app.exceptions import ValidationError
from . import db, login_manager
from datetime import datetime

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S'


class Permission:
    FOLLOW = 0x01
    COMMENT = 0x02
    WRITE_ARTICLES = 0x04
    MODERATE_COMMENTS = 0x08
    ADMINISTER = 0x80


class CRUDMixin(object):
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)

    @classmethod
    def get_by_id(cls, id):
        if any(
                (isinstance(id, basestring) and id.isdigit(),
                 isinstance(id, (int, float))),
        ):
            return cls.query.get(int(id))
        return None

    @classmethod
    def create(cls, **kwargs):
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        for attr, value in kwargs.iteritems():
            setattr(self, attr, value)
        return commit and self.save() or self

    def save(self, commit=True):
        db.session.add(self)
        if commit:
            db.session.commit()
        return self

    def delete(self, commit=True):
        db.session.delete(self)
        return commit and db.session.commit()


class Role(CRUDMixin, db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    default = db.Column(db.Boolean, default=False, index=True)
    permissions = db.Column(db.Integer)
    users = db.relationship('User', backref='role', lazy='dynamic')

    @staticmethod
    def insert_roles():
        roles = {
            'User': (Permission.FOLLOW |
                     Permission.COMMENT |
                     Permission.WRITE_ARTICLES, True),
            'Moderator': (Permission.FOLLOW |
                          Permission.COMMENT |
                          Permission.WRITE_ARTICLES |
                          Permission.MODERATE_COMMENTS, False),
            'Administrator': (0xff, False)
        }
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            role.permissions = roles[r][0]
            role.default = roles[r][1]
            db.session.add(role)
        db.session.commit()

    def __repr__(self):
        return '<Role %r>' % self.name


class Follow(CRUDMixin, db.Model):
    __tablename__ = 'follows'
    follower_id = db.Column(db.Integer, db.ForeignKey('users.id'),
                            primary_key=True)
    followed_id = db.Column(db.Integer, db.ForeignKey('users.id'),
                            primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class User(CRUDMixin, UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    password_hash = db.Column(db.String(128))
    confirmed = db.Column(db.Boolean, default=False)
    name = db.Column(db.String(64))
    location = db.Column(db.String(64))
    about_me = db.Column(db.Text())
    member_since = db.Column(db.DateTime(), default=datetime.utcnow)
    last_seen = db.Column(db.DateTime(), default=datetime.utcnow)
    avatar_hash = db.Column(db.String(32))
    posts = db.relationship('Post', backref='author', lazy='dynamic')
    followed = db.relationship('Follow',
                               foreign_keys=[Follow.follower_id],
                               backref=db.backref('follower', lazy='joined'),
                               lazy='dynamic',
                               cascade='all, delete-orphan')
    followers = db.relationship('Follow',
                                foreign_keys=[Follow.followed_id],
                                backref=db.backref('followed', lazy='joined'),
                                lazy='dynamic',
                                cascade='all, delete-orphan')
    comments = db.relationship('Comment', backref='author', lazy='dynamic')

    @staticmethod
    def generate_fake(count=100):
        from sqlalchemy.exc import IntegrityError
        from random import seed
        import forgery_py

        seed()
        for i in range(count):
            u = User(email=forgery_py.internet.email_address(),
                     username=forgery_py.internet.user_name(True),
                     password=forgery_py.lorem_ipsum.word(),
                     confirmed=True,
                     name=forgery_py.name.full_name(),
                     location=forgery_py.address.city(),
                     about_me=forgery_py.lorem_ipsum.sentence(),
                     member_since=forgery_py.date.date(True))
            db.session.add(u)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()

    @staticmethod
    def add_self_follows():
        for user in User.query.all():
            if not user.is_following(user):
                user.follow(user)
                db.session.add(user)
                db.session.commit()

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.role is None:
            if self.email == current_app.config['FLASKY_ADMIN']:
                self.role = Role.query.filter_by(permissions=0xff).first()
            if self.role is None:
                self.role = Role.query.filter_by(default=True).first()
        if self.email is not None and self.avatar_hash is None:
            self.avatar_hash = hashlib.md5(
                self.email.encode('utf-8')).hexdigest()
        self.followed.append(Follow(followed=self))

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_confirmation_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'confirm': self.id})

    def confirm(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('confirm') != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True

    def generate_reset_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'reset': self.id})

    def reset_password(self, token, new_password):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('reset') != self.id:
            return False
        self.password = new_password
        db.session.add(self)
        return True

    def generate_email_change_token(self, new_email, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'change_email': self.id, 'new_email': new_email})

    def change_email(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return False
        if data.get('change_email') != self.id:
            return False
        new_email = data.get('new_email')
        if new_email is None:
            return False
        if self.query.filter_by(email=new_email).first() is not None:
            return False
        self.email = new_email
        self.avatar_hash = hashlib.md5(
            self.email.encode('utf-8')).hexdigest()
        db.session.add(self)
        return True

    def can(self, permissions):
        return self.role is not None and \
               (self.role.permissions & permissions) == permissions

    def is_administrator(self):
        return self.can(Permission.ADMINISTER)

    def ping(self):
        self.last_seen = datetime.utcnow()
        db.session.add(self)

    def gravatar(self, size=100, default='identicon', rating='g'):
        if request.is_secure:
            url = 'https://secure.gravatar.com/avatar'
        else:
            url = 'http://www.gravatar.com/avatar'
        hash = self.avatar_hash or hashlib.md5(
            self.email.encode('utf-8')).hexdigest()
        return '{url}/{hash}?s={size}&d={default}&r={rating}'.format(
            url=url, hash=hash, size=size, default=default, rating=rating)

    def follow(self, user):
        if not self.is_following(user):
            f = Follow(follower=self, followed=user)
            db.session.add(f)

    def unfollow(self, user):
        f = self.followed.filter_by(followed_id=user.id).first()
        if f:
            db.session.delete(f)

    def is_following(self, user):
        return self.followed.filter_by(
            followed_id=user.id).first() is not None

    def is_followed_by(self, user):
        return self.followers.filter_by(
            follower_id=user.id).first() is not None

    @property
    def followed_posts(self):
        return Post.query.join(Follow, Follow.followed_id == Post.author_id) \
            .filter(Follow.follower_id == self.id)

    def to_json(self):
        json_user = {
            'url': url_for('api.get_post', id=self.id, _external=True),
            'username': self.username,
            'member_since': self.member_since,
            'last_seen': self.last_seen,
            'posts': url_for('api.get_user_posts', id=self.id, _external=True),
            'followed_posts': url_for('api.get_user_followed_posts',
                                      id=self.id, _external=True),
            'post_count': self.posts.count()
        }
        return json_user

    def generate_auth_token(self, expiration):
        s = Serializer(current_app.config['SECRET_KEY'],
                       expires_in=expiration)
        return s.dumps({'id': self.id}).decode('ascii')

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        return User.query.get(data['id'])

    def __repr__(self):
        return '<User %r>' % self.username


class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False

    def is_administrator(self):
        return False


login_manager.anonymous_user = AnonymousUser


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Post(CRUDMixin, db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text)
    body = db.Column(db.Text)
    body_html = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    author_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    node_id = db.Column(db.Integer, db.ForeignKey('som_nodes.id'))
    comments = db.relationship('Comment', backref='post', lazy='dynamic')
    topics = db.relationship('TopicPost', backref='post', lazy='dynamic')

    @staticmethod
    def generate_fake(count=100):
        from random import seed, randint
        import forgery_py

        seed()
        user_count = User.query.count()
        for i in range(count):
            u = User.query.offset(randint(0, user_count - 1)).first()
            p = Post(body=forgery_py.lorem_ipsum.sentences(randint(1, 5)),
                     timestamp=forgery_py.date.date(True),
                     author=u)
            db.session.add(p)
            db.session.commit()

    @staticmethod
    def on_changed_body(target, value, oldvalue, initiator):
        allowed_tags = ['a', 'abbr', 'acronym', 'b', 'blockquote', 'code',
                        'em', 'i', 'li', 'ol', 'pre', 'strong', 'ul',
                        'h1', 'h2', 'h3', 'p']
        target.body_html = bleach.linkify(bleach.clean(
            markdown(value, output_format='html'),
            tags=allowed_tags, strip=True))

    def to_json(self):
        json_post = {
            'url': url_for('api.get_post', id=self.id, _external=True),
            'body': self.body,
            'body_html': self.body_html,
            'timestamp': self.timestamp,
            'author': url_for('api.get_user', id=self.author_id,
                              _external=True),
            'comments': url_for('api.get_post_comments', id=self.id,
                                _external=True),
            'comment_count': self.comments.count()
        }
        return json_post

    @staticmethod
    def from_json(json_post):
        body = json_post.get('body')
        title = json_post.get('title')
        if body is None or body == '':
            raise ValidationError('post does not have a body')
        if title is None or title == '':
            raise ValidationError('post does not have a title')
        return Post(body=body, title=title)

    @staticmethod
    def from_dataframe(df):
        """Reads the content of a dataframe and generate a list of Post
        items with title, body and timestamp.
        :type df: pandas.DataFrame
        """
        posts = []
        for index, post in df.iterrows():
            # Todo: import the author field
            body = df["body"]
            title = df["title"]
            timestamp = datetime.strptime(df["timestamp"].split("+")[0],
                                          TIMESTAMP_FORMAT)
            db.session.add(Post(body=body, title=title, timestamp=timestamp))
        return posts

    @staticmethod
    def on_change_node(df, json_context):
        """Change the node ID of the post
        """
        context = Context.from_json(json_context)
        db.session.merge(context)
        for index, node in df.iterrows():
            node = db.session.query(Node).filter(Node.x == df["x"],
                                                 Node.x == df["y"],
                                                 Node.context == context).first()
            if node:
                node.node = node
                db.session.add(node)
        db.session.commit()


db.event.listen(Post.body, 'set', Post.on_changed_body)


class Comment(CRUDMixin, db.Model):
    __tablename__ = 'comments'
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text)
    body_html = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    disabled = db.Column(db.Boolean)
    author_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'))

    @staticmethod
    def on_changed_body(target, value, oldvalue, initiator):
        allowed_tags = ['a', 'abbr', 'acronym', 'b', 'code', 'em', 'i',
                        'strong']
        target.body_html = bleach.linkify(bleach.clean(
            markdown(value, output_format='html'),
            tags=allowed_tags, strip=True))

    def to_json(self):
        json_comment = {
            'url': url_for('api.get_comment', id=self.id, _external=True),
            'post': url_for('api.get_post', id=self.post_id, _external=True),
            'body': self.body,
            'body_html': self.body_html,
            'timestamp': self.timestamp,
            'author': url_for('api.get_user', id=self.author_id,
                              _external=True),
        }
        return json_comment

    @staticmethod
    def from_json(json_comment):
        body = json_comment.get('body')
        if body is None or body == '':
            raise ValidationError('comment does not have a body')
        return Comment(body=body)


db.event.listen(Comment.body, 'set', Comment.on_changed_body)


class TopicPost(CRUDMixin, db.Model):
    __tablename__ = "topic_post"
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'))
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'))
    weight = db.Column(db.Float)


class TopicNode(CRUDMixin, db.Model):
    __tablename__ = 'topic_node'
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'))
    node_id = db.Column(db.Integer, db.ForeignKey('som_nodes.id'))
    weight = db.Column(db.Float)


class Topic(CRUDMixin, db.Model):
    __tablename__ = "topics"
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text)
    posts = db.relationship('TopicPost', backref='topic', lazy='dynamic')
    nodes = db.relationship('TopicNode', backref='topic', lazy='dynamic')

    @staticmethod
    def from_dataframe(topics_df):
        # Todo : import the topics from a pandas df
        topics = []
        return None

    @staticmethod
    def from_json(json_topic):
        # Todo import topics from json
        return None

    def to_json(self):
        json_topic = {
            'url': url_for('api.get_topic', id=self.id, _external=True),
            'data': jsonify(self.data),
            'posts': url_for('api.get_topic_posts', id=self.id, _external=True),
            'nodes': url_for('api.get_topic_nodes', id=self.id, _external=True),
        }
        return json_topic


class Node(CRUDMixin, db.Model):
    __tablename__ = "som_nodes"
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    posts = db.relationship('Post', backref='node', lazy='dynamic')
    topics = db.relationship('TopicNode', backref='node', lazy='dynamic')
    cluster_id = db.Column(db.Integer, db.ForeignKey('som_clusters.id'))
    context_id = db.Column(db.Integer, db.ForeignKey('context.id'))

    @staticmethod
    def from_dataframe(df, json_context):
        # Main import function: also appends the context data
        nodes = []
        session = db.session
        context = Context.from_json(jsonify(json_context))
        session.merge(context)
        for index, node in df.iterrows():
            cluster = Cluster(id=int(node["cluster"]))
            session.merge(cluster)
            x = int(node["x"])
            y = int(node["y"])
            nodes.append(Node(x=x, y=y, cluster=cluster, context=context))
        return nodes

    @staticmethod
    def from_json(json_node):
        x = json_node.get("x")
        y = json_node.get("y")
        cluster_id = json_node.get("cluster")
        cluster = Cluster(id=cluster_id)

        return Node(x=x, y=y, cluster=cluster)

    def to_json(self):
        json_node = {
            'url': url_for('api.get_node', id=self.id, context_id=self.context_id, _external=True),
            'x': self.x,
            'y': self.y,
            'posts': url_for('api.get_node_posts', id=self.id, _external=True),
            'topics': url_for('api.get_node_topics', id=self.id, _external=True),
            'cluster': url_for('api.get_cluster', id=self.cluster_id, _external=True),
            'context': url_for('api.get_context', id=self.context_id, _external=True)
        }
        return json_node


class Cluster(CRUDMixin, db.Model):
    __tablename__ = "som_clusters"
    id = db.Column(db.Integer, primary_key=True)
    nodes = db.relationship('Node', backref='cluster', lazy='dynamic')

    def to_json(self):
        json_cluster = {
            'url': url_for('api.get_cluster', id=self.id, _external=True),
            'nodes': url_for('api.get_cluster_nodes', id=self.id, _external=True),
        }
        return json_cluster


class Context(CRUDMixin, db.Model):
    __tablename__ = 'context'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    x_shape = db.Column(db.Integer)
    y_shape = db.Column(db.Integer)
    n_clusters = db.Column(db.Integer)
    nodes = db.relationship('Node', backref='context', lazy='dynamic')

    def to_json(self):
        json_context = {
            'timestamp': datetime.strftime(TIMESTAMP_FORMAT),
            'x_shape': self.x_shape,
            'y_shape': self.y_shape,
            'n_clusters': self.n_clusters,
            'nodes': url_for('api.get_context_nodes', id=self.id, _external=True)
        }
        return json_context

    @staticmethod
    def from_json(context_json):
        timestamp = datetime.strptime(context_json.get('timestamp'), TIMESTAMP_FORMAT)
        x_shape = context_json.get('x_shape')
        y_shape = context_json.get('y_shape')
        n_clusters = context_json.get('n_clusters')

        if x_shape is None or x_shape == '':
            raise ValidationError('Context does not have x_shape')
        if y_shape is None or y_shape == '':
            raise ValidationError('Context does not have y_shape')
        if n_clusters is None or n_clusters == '':
            raise ValidationError('Context does not have cluster no')

        return Context(timestamp=timestamp, x_shape=x_shape,
                       y_shape=y_shape, nclusters=n_clusters)
