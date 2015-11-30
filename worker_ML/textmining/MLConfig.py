__author__ = 'husser'

PARAMETERS_VECTORIZE = {
    "vocabulary": None,
    "max_df": 0.6,
    "min_df": 0.01,
    # "analyzer": 'word',
    "ngram_range": (1, 1),  # unigrams or bigrams
    # "token_pattern":"([a-zA-Z]+)",
    "encoding": "utf-8",
    "strip_accents": 'ascii',
    "norm": 'l2',
}

PARAMETERS_LSA = {
    "n_components": 300,
    "algorithm": "arpack",
    # "n_iter": 300,
    # "random_state":42,
}

PARAMETERS_NMF = {
    "n_components": 200,
}

PARAMETERS_LDA = {
    "n_topics": 200,
    "n_iter": 1,
    # "doc_topic_prior": 0.1, #known as alpha
    # "topic_word_prior": 0.001, #known as eta
    # "learning_method":"batch",
}

PARAMETERS_SOM = {
    "kshape": (9, 16),
    "n_iter": 250,
    "learning_rate": 0.005,
    "initialization_func": None,
    "topology": "rect"
}

PARAMETERS_WARD = {
    'n_clusters': 6,
    'linkage': 'ward',
}
