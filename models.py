from datetime import datetime, timedelta
from itertools import combinations
import math as math
from os import defpath
import pickle
import re
import warnings
import logging

import boto3
import nltk
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy import spatial

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(level = logging.INFO)
warnings.filterwarnings('ignore')


def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)


def pairwise_cosine_similarity(x):
    return [cosine_similarity(pair[0], pair[1]) for pair in combinations(x, 2)]


class ClusterModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y=None):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def fit_predict(self, X, y=None):
        return self.model.fit_predict(X, y)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)
        
    def get_n_clusters(self):
        try:
            return self.model.n_clusters
        except(AttributeError):
            return self.model.cluster_centers_.shape[0]
        

class Node:
    def __init__(self, depth, mask, center):
        self.depth = depth
        self.mask = mask
        self.center = center
        self.children = []
        
    def add_child(self, mask, center):
        child = Node(self.depth+1, mask, center)
        self.children.append(child)
        return child
        
        
class TopicTreeModel:
    def __init__(self, config):
        self.config = config
        self.root = None
        
    @staticmethod   
    def fit_model(X, mask, model_spec):
        model_func, model_params = model_spec
        model = ClusterModelWrapper(model_func(**model_params))
        labels = model.fit_predict(X[mask])
        padded = -1 * np.ones(X.shape[0])
        padded[mask] = labels
        return padded, model.model.cluster_centers_
    
    def fit_level(self, X, parent):
        if parent.depth == len(self.config):
            return
        
        labels, centers = self.fit_model(X, parent.mask, self.config[parent.depth])
        print(np.unique(labels))
        
        for cluster in range(int(max(labels))+1):
            print(cluster)
            cluster_mask = (labels == cluster) * parent.mask
            cluster_center = centers[cluster]
            node = parent.add_child(mask=cluster_mask, center=cluster_center)
            
            self.fit_level(X, node)
            
    def fit(self, X):
        self.root = Node(depth=0, mask=np.ones(X.shape[0], dtype=bool), center=None)
        self.fit_level(X, self.root)
