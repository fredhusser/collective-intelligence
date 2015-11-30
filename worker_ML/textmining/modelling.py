# ===============================================================================
# for python 2.7
# Author: Frederic Husser
# Filename: modelling.py
# Description:
# The application manager is used for managing the asynchrone data indexing
# sessions given a set of serialized records (posts) and some session settings.
# The manager is responsible for the connection with the remote data store
# MongoDB and the data repository.
# ===============================================================================


import numpy as np
# Import all the required algorithms for the models
from textmining.som import SOMMapper
from nltk.stem.snowball import FrenchStemmer
from nltk import word_tokenize
# from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

DATABASE_CONNECT = "sqlite:///../data/scrape.sqlite"
stemmer = FrenchStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class Model(object):
    """Basic implementation of the class interface, that follows the 
    requirements of the Scikit-Learn framework, and adds IO and metrics
    features. Models classes can be assembled together and provide three 
    principal data interfaces:
        - The model data transformation the distribution from the input s
        pace into the output space.
        - The model prediction giving the assignment of input data samples
        into the output space.
        - The model parameters are used for the fitting process.
    Different models can be stacked together in the way of a pipeline.
    """
    model_name = "sample_model"
    output_data_name = "output_data"
    mapper_name = "mapper_name"
    model_type = "model"

    def __init__(self, input_data, fit_parameters, output_space_size=0):
        self.input_data = input_data
        self.fit_parameters = fit_parameters
        self.training_set_size = input_data.shape[0]
        self.output_space_size = output_space_size
        self.model_attributes = {}

    def fit(self):
        """Function to run the model on the input data. Override this
        function to implement your custom mapper.
        It must implement the computation of:
            - the output_data array, shape = [n_output, n_input]
            - the mapper array, shape = [n_samples]
        """
        self.output_data = np.array([])
        self.mapper_data = np.array
        return self

    def transform(self):
        return self.output_data

    def predict(self):
        return self.mapper

    def log(self, msg):
        print(self.model_name + ":\t" + msg)

    def _log_model_results(self):
        self.log("Model %s of type %s was trained" % (self.model_name,
                                                      self.model_type))
        self.log("Output name:\t %s" % self.output_data_name)
        self.log("Input space:\t %s" % str(self.input_data.shape))
        self.log("Output space:\t %s" % str(self.output_data.shape))
        self.log("Model attributes:")
        for key, value in self.model_attributes.iteritems():
            self.log("\t%s : \t%s" % (key, value))



class ModelTFIDF(Model):
    """Class to implement the TF-IDF vectorizing 
    """
    model_type = "transformer"
    model_name = "tfidf"
    output_data_name = "tfidfMatrix"
    mapper_name = "words_to_index"

    def fit(self, language=None):
        self._set_tokenizer(language)
        vectorizer = TfidfVectorizer(**self.fit_parameters)

        self.output_data = vectorizer.fit_transform(self.input_data)
        self.output_space_size = self.output_data.shape[1]

        # Give access to the index of a word
        self.mapper_data = vectorizer.vocabulary_

        # Return a reverse mapper for retrieving a word from index
        self.mapper_reverse = np.empty(self.output_space_size, dtype="a30")
        for key, value in self.mapper_data.iteritems():
            self.mapper_reverse[value] = key

        self._log_model_results()
        return self

    def _set_tokenizer(self, language):
        def stem_tokens(tokens):
            stemmed = []
            for item in tokens:
                stemmed.append(stemmer.stem(item))
            return stemmed

        def tokenizer(text):
            tokens = word_tokenize(text)
            stems = stem_tokens(tokens)
            return stems

        if language == "french":
            stemmer = FrenchStemmer()
            self.fit_parameters["tokenizer"] = tokenizer


class ModelLSA(Model):
    """Class to implement the dimensionality reduction of the raw
    Vector Space Model of the documents stored as a sparse matrix 
    of TF-IDF values.
    Reduces the dimensionality from n_features to n-components based 
    on the latent sementic indexing.
    """
    model_type = "transformer"
    model_name = "lsa"
    output_data_name = "docComponentsMatrix"
    mapper_name = "words_to_topics"

    def fit(self):
        if self.output_space_size == 0:
            self.output_space_size = self.fit_parameters["n_components"]
        else:
            self.fit_parameters["n_components"] = self.output_space_size
        svd = TruncatedSVD(**self.fit_parameters)
        lsa = make_pipeline(svd, Normalizer(copy=False))

        self.output_data = svd.fit_transform(self.input_data)
        self.output_data = Normalizer().fit_transform(self.output_data)

        self.mapper_data = svd.components_
        self.model_attributes = {"n_components": self.output_space_size,
                                 "n_iter": svd.n_iter,
                                 "algorithm": svd.algorithm,
                                 }
        self._log_model_results()
        return self


@NotImplementedError
class ModelLDA(Model):
    """Class to implement the dimensionality reduction of the raw
    Vector Space Model of the documents, stored as a large and sparse
    TF-IDF matrix into a limited set of topic
    This is a bayesian version of the latent semantic indexing.
    """
    model_type = "transformer"
    model_name = "lda"
    output_data_name = "docTopicsMatrix"
    mapper_name = "words_to_topics"

    def fit(self):
        lda_reduce = LDA(**self.fit_parameters)

        self.output_data = lda_reduce.fit_transform(self.input_data)
        self.mapper_data = lda_reduce.components_
        self.model_attributes = {"n_topics": self.output_space_size,
                                 "n_iter": lda_reduce.n_iter_,
                                 "alpha": lda_reduce.doc_topic_prior,
                                 "eta": lda_reduce.topic_word_prior,
                                 }
        self._log_model_results()
        return self


class ModelNMF(Model):
    """Class to implement the dimensionality reduction of the raw
    Vector Space Model of the documents, stored as a large and sparse
    TF-IDF matrix into a limited set of topics
    Using the Non-negative Matrix Factorization it is first way to 
    group words into topics. It is not as elaborate as the Latent 
    Dirichlet Allocation but a more interpretable output than LSI
    """
    model_type = "transformer"
    model_name = "nmf"
    output_data_name = "docTopicsMatrix"
    mapper_name = "words_to_topics"

    def fit(self):
        nmf = NMF(**self.fit_parameters)
        nmf.fit(self.input_data)

        self.output_data = nmf.transform(self.input_data)
        self.mapper_data = nmf.components_
        self.model_attributes = {"n_topics": nmf.n_components,
                                 }
        self._log_model_results()
        return self


class ModelSOM(Model):
    """Class to implement the modeling of the SOM mapper. From
    assigning documents represented in the topic space into a 
    2-dimensional space, preserving the topology.
    """
    model_type = "mapper"
    model_name = "som"
    output_data_name = "kohonenMatrix"
    mapper_name = "articles_to_nodes"

    def fit(self, K_init=None):
        self.log("Vectorized Corpus of size:")
        self.log("%d samples and %d features" % self.input_data.shape)

        # Check consistency of the required output space size
        kshape = self.fit_parameters["kshape"]
        n_nodes = kshape[0] * kshape[1]
        if self.output_space_size != n_nodes:
            self.output_space_size = n_nodes
            self.log("Output Shape not matching")

        som_mapper = SOMMapper(**self.fit_parameters)
        som_mapper.fit(self.input_data, K_init=K_init)

        # Data Creation
        self.output_data = som_mapper.transform()
        self.mapper_data = som_mapper.predict(self.input_data)
        self.model_attributes = {"kshape": som_mapper.kshape,
                                 "n_nodes": self.output_space_size,
                                 "topology": som_mapper.topology,
                                 "learning_rate": som_mapper.learning_rate,
                                 "quant_error": som_mapper.quantization_error[-1],
                                 "topo_error": som_mapper.topological_error}
        self._log_model_results()
        return self


class ModelKMeans(Model):
    """Implement the final clustering of the SOM map seen as distribution
    of documents with a topology preserving mapping on a 2D set. This 
    unsupervised clustering technique helps the user visualizing a reduced
    number of groups of articles to be more appealing visually.
    """
    model_type = "clustering"
    model_name = "kmeans"
    output_data_name = "clustersMatrix"
    mapper_name = "articles_to_nodes"

    def fit(self):
        kmeans = KMeans(**self.fit_parameters)
        kmeans.fit(self.input_data)
        self.mapper_data = kmeans.predict(self.input_data)
        self.output_data = kmeans.cluster_centers_
        self.output_space_size = kmeans.n_clusters
        self.model_attributes = {"n_clusters": kmeans.n_clusters,
                                 "n_iter": kmeans.n_iter_,
                                 "inertia": kmeans.inertia_}
        self._log_model_results()
        return self


class ModelWARD(Model):
    """Aglomerative Clustering allows for structured clustering of data
    when topologically structured in a 2D/3D space. The clustering is 
    however performed against higher dimensional feature space.

    Reference
    ---------

    Scikit-Learn: http://scikit-learn.org/stable/modules/clustering.html#clustering

    """
    model_type = "clustering"
    model_name = "ward"
    output_data_name = "None"
    mapper_name = "articles_to_nodes"

    def fit(self, kshape=None):
        if kshape is not None:
            connectivity = grid_to_graph(*kshape)
            self.fit_parameters.update({"connectivity": connectivity})
        ward = AgglomerativeClustering(**self.fit_parameters)
        ward.fit(self.input_data)
        self.mapper_data = ward.labels_
        self.output_data = np.array([])
        self.output_space_size = ward.n_clusters
        self.model_attributes = {"n_clusters": ward.n_clusters,
                                 "n_components": ward.n_components}
        self._log_model_results()
        return self





def log(msg):
    # To do give options to the logging functionality
    print(msg)

