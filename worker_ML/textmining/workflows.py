"""File To manage the workflows dealing with the data analysis tasks
so that preconfigured pipelines can be used in servers such as Flask
for fetching data, performing analysis tasks and rendering the results.
"""

__author__ = 'husser'

import numpy as np
import pandas as pd
import json
from bson import json_util
from bson.json_util import dumps
from modelling import ModelTFIDF, ModelLSA, ModelNMF, ModelSOM, ModelWARD
from .MLConfig import *


def data_frame(query, columns):
    """
    Takes a sqlalchemy query and a list of columns, returns a dataframe.
    """

    def make_row(x):
        return dict([(c, getattr(x, c)) for c in columns])

    return pd.DataFrame([make_row(x) for x in query])


class Corpus(object):
    """Implement the organizing of the corpus from a given source. It fetches
    the documents and eventually offers a direct interface as a numpy array
    and as an iterator. The last ensures that streaming of the input data is
    possible when exceeding the RAM capacity

    Parameters:
    -----------
    source: file-like
        where to fetch the dat
    type: string,
        Must in 'sqlite', 'csv', 'hdf5', 'url' determines the way the data
        is fetched
    """


class SemanticAnalysisWorkflow(object):
    """Basic workflow class for performing data analysis
    it overrides the Corpus class in order to perform some
    analysis on it. It provides an analysis
    """
    MONGO_DBNAME_OUT = "textmining"
    MONGO_COLLECTION_OUT = "semantic"

    def __init__(self, connection, dbname, collection):
        self.attributes = {}
        self.vectorize_ = None
        self.reduce_ = None
        self.classify_ = None
        self.cluster_ = None
        self.connection = connection
        self.read_mongo(dbname,collection)

    def read_mongo(self, dbname, collection):
        """Data extractor from a mongo DB client
        """
        collection = self.connection[dbname][collection]
        articles = collection.find(projection={"title": True, "body": True, "_id": False})

        json_articles = []
        for article in articles:
            json_articles.append(article)
        json_data = json.dumps(json_articles, default=json_util.default)
        self.corpus_data = pd.read_json(json_data).drop_duplicates().dropna()
        print self.corpus_data.head()
        return self

    def read_sql(self, session, post_model, size_limit=None):
        """Data extractor from a SQLAlchemy session. Must
        provide a corresponding model for extracting the posts.
        Todo: Provide a connector to a web api
        :param session:
        :param post_model:
        :return: self
        """
        if size_limit != None:
            query = session.query(post_model).limit(size_limit)
        else:
            query = session.query(post_model)
        self.corpus_data = data_frame(query, columns=["title", "body"])
        return self

    def log(self, msg):
        print(msg)

    def main(self):
        """Used as a main interface to the workflow class
        :param reducer:
        :return:
        """
        self.vectorizer().reducer().classifier()
        self.aggregate_map_data(n_topics=4, n_features=5)
        self.store_in_mongo(self.MONGO_DBNAME_OUT,self.MONGO_COLLECTION_OUT)

    def vectorizer(self):
        """Representation of the text data into the vector space model.
        """
        self.vectorize_ = ModelTFIDF(self.corpus_data.body, PARAMETERS_VECTORIZE).fit()
        self.attributes["corpus_size"] = self.vectorize_.training_set_size
        self.attributes["n_features"] = self.vectorize_.output_space_size
        return self

    def reducer(self):
        """Run the semantic analysis for topic modelling prior to data analysis
        """
        self.reduce_ = ModelLSA(self.vectorize_.output_data,
                                    PARAMETERS_LSA).fit()

        self.attributes["n_topics"] = self.reduce_.output_space_size
        return self

    def classifier(self):
        """Sample pipeline for text vectorizing, reduction, classification
        and clustering
        """
        # SOM Mapper
        self.classify_ = ModelSOM(self.reduce_.output_data,
                                  PARAMETERS_SOM).fit()
        self.attributes["n_nodes"] = self.classify_.output_space_size
        self.attributes["kshape"] = self.classify_.model_attributes["kshape"]

        # Clustering the Feature map
        self.cluster_ = ModelWARD(self.classify_.output_data,
                                  PARAMETERS_WARD).fit(self.classify_.model_attributes["kshape"])
        self.attributes["n_clusters"] = self.cluster_.model_attributes["n_clusters"]
        return self

    ### Methods for data retrieval ###
    def getTokensByTopic(self, topic_id, n_words):
        """Get the top features for a given topic
        Return
            json_data: [{"name":"string",
                         "weight":"float",
                         "rank":"integer"}]
        """
        n_features = self.attributes["n_features"]
        features_ids = np.argsort(self.reduce_.mapper_data[topic_id, :])[:n_features - n_words - 1:-1]
        return [{"name": self.vectorize_.mapper_reverse[token],
                 "weight": self.reduce_.mapper_data[topic_id, token],
                 "rank": int(index),
                 } for index, token in enumerate(features_ids)]

    def getTokensByNode(self, node_id, max_features, max_topics):
        """Chaining the functions getTopicsByNode and
        getTokensByTopic.

        Return:
            json_data: [{"rank":"integer",
                         "weight":"float",
                         "id": "integer",
                         "tokens":[{"token":"string",
                                    "frequency":"float"}]
        """
        topics_collection = self.getTopicsIdByNode(node_id, max_topics)
        for topic_item in topics_collection:
            topic_item["tokens"] = self.getTokensByTopic(topic_item["topic_id"],
                                                         max_features)
        return topics_collection

    def getTopicsIdByNode(self, node_id, topics):
        """Give the top topics ids of a given node of the
        Kohonen map.

        Return:
            json = [{"rank":"integer",
                    "weight":"float",
                    "topic_id":"index"}]
        """
        n_topics = self.attributes["n_topics"]
        ids = np.argsort(self.classify_.output_data[node_id, :])
        sorted_ = ids[:n_topics - topics - 1:-1]
        return [{"rank": int(index),
                 "weight": self.classify_.output_data[node_id, topic_id],
                 "topic_id": int(topic_id)} for index, topic_id in enumerate(sorted_)]

    ### Retrieve SOM Nodes ###
    def getNodeIdByCluster(self, cluster_id):
        """ Gives for a given cluster the list of nodes.
        """
        return np.where(self.cluster_.mapper_data == cluster_id)[0]

    ### Retrieve Articles ###
    def getArticlesIdByTopic(self, topic_id, n_articles):
        """Get the top articles per topic
        """
        corpus_size = self.attributes["corpus_size"]
        ids = np.argsort(self.reduce_.output_data[:, topic_id])
        return ids[:corpus_size - n_articles - 1:-1]

    def getArticlesIdByNode(self, node_id):
        """Get the list of articles mapped into a node of the SOM.
        This is strictly to say based on the mapper_data
        """
        return np.where(self.classify_.mapper_data == node_id)[0]

    def getArticlesIdByCluster(self, cluster_id):
        """Chaining the function getArticlesIdByNode and
        getNodeIDByCluster to group the set of articles together
        """
        nodes = self.getNodeIdByCluster(cluster_id)
        articles = []
        for node_id in nodes:
            articles.append(self.getArticlesIdByNode(node_id))

        if articles != []:
            articles = np.concatenate(articles)
        else:
            articles = np.array([])
        return articles

    # Methods for data aggregation
    def aggregate_corpus_data(self, n_topics=3, n_features=5):
        """Create a dataframe that summarizes the corpus data as mapped through
        the analysis. Each document is represented by its id, title, node coordinates
        and cluster. The top topics of the vectorized document are serialized into
        strings.
        """
        df = self.corpus_data
        n_cols = self.attributes["kshape"][1]
        index = self.corpus_data.index.values
        # df.body = df.body.s
        # df.title = df.title.str.encode("utf8")

        df["node"] = self.classify_.mapper_data[index]
        df["x"] = np.divide(index, n_cols).astype('int')
        df["y"] = index % n_cols
        df["cluster"] = self.cluster_.mapper_data[df["node"].ix[index]]
        df["topics"] = np.array([json.dumps(self.getTokensByNode(
            df["node"].ix[ix],
            n_features,
            n_topics)) for ix in index])
        return df

    def aggregate_map_data(self, n_topics=2, n_features=5):
        """Create a dataframe that summarizes the map data for each node
        with the node coordinates, the cluster id, the main topics described
        by the top features and the number of articles contained in the map.
        The articles must be retrieved from the documents database.
        """
        n_nodes = self.attributes["n_nodes"]
        nodes = np.arange(n_nodes)
        n_cols = self.attributes["kshape"][1]
        data = np.column_stack((nodes,
                                np.divide(nodes, n_cols),
                                nodes % n_cols,
                                self.cluster_.mapper_data[nodes]))
        self.nodes_data = pd.DataFrame(data, columns=["node", "x", "y", "cluster"])
        for topic in xrange(n_topics):
            self.nodes_data["topic_%d" % (1 + topic)] = \
                json.dumps(self.getTokensByNode(topic, n_features, n_topics))
        self.nodes_data["hits"] = self.nodes_data.groupby("node").size()
        self.nodes_data.fillna(0, inplace=True)
        return self.nodes_data

    def print_featuresByTopic(self, topic_id, n_features):
        tokens = self.getTokensByTopic(topic_id, 10)
        for i in xrange(len(tokens)):
            self.log("%s:\t %.4f" % (tokens[i]["name"], tokens[i]["weight"]))

    def print_articlesTitlesByTopic(self, topic_id, n_articles):
        for doc_id in self.getArticlesIdByTopic(topic_id, n_articles):
            self.log("%d\t %s" % (doc_id, self.corpus_data.title[doc_id]))

    def store_in_mongo(self, db_name, collection):
        """Function to store the data in the Mongo DB for use as data source for
        visualization
        :param db_name: Database for the subproject
        :param collection: Current collection in which data is stored
        """
        from datetime import datetime
        collection = self.connection[db_name][collection]

        # Hierarchical Data Base
        session = {
            "timestamp": datetime.utcnow(),
            "n_clusters": self.attributes["n_clusters"],
            "k_shape": {"x_shape": self.attributes["kshape"][0],
                        "y_shape": self.attributes["kshape"][1]},
            "n_nodes": self.attributes["n_nodes"],
            "n_topics": self.attributes["n_topics"],
            "nodes": [],
            "topics": [],
        }
        n_nodes = self.attributes["n_nodes"]
        n_topics = self.attributes["n_topics"]
        # Add the nodes data
        for node_id in xrange(n_nodes):
            coords = (int(np.divide(node_id, self.attributes["kshape"][1])),
                      int(node_id % self.attributes["kshape"][1]))
            articles = self.getArticlesIdByNode(node_id=node_id)
            topics = self.getTokensByNode(node_id=node_id, max_features=10, max_topics=5)
            node = {
                "id": node_id,
                "coordinates": {"x_shape": coords[0],
                                "y_shape": coords[1]},
                "article_hits": len(articles),
                "articles": articles.tolist(),
                "topics": topics,
            }
            session["nodes"].append(node)

        # Add the topics data
        for topic_id in xrange(n_topics):
            topic = {
                "id": topic_id,
                "tokens": self.getTokensByTopic(topic_id=topic_id, n_words=10)
            }
            session["topics"].append(topic)

        collection.insert_one(session)


class Workflow_NMF(SemanticAnalysisWorkflow):
    """Implementation of the semantic analysis with the Non-negative matrix factorization
    approach instead of the latent semantic analysis
    """
    def reducer(self):
        self.reduce_ = ModelNMF(self.vectorize_.output_data,
                        PARAMETERS_NMF).fit()
        self.attributes["n_topics"] = self.reduce_.output_space_size
        return self
