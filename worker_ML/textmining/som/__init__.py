"""
The :mod:`NeuralNetworks.app.core.som` module implements a self-organizing map
algorithm for classification of text documents. This follows the guidelines
and interfaces of the Scikit-Learn framework.
"""

from .som import SOMMapper, build_U_matrix

__all__ = ['SOMMapper', 'build_U_matrix']

def debug(alg, string):
    print(alg,string)