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
    def __init__(self, config, n_candidates=10, n_return=3):
        self.config = config
        self.root = None
        self.n_candidates = n_candidates
        self.n_return = n_return
        
    @staticmethod   
    def fit_model(X, mask, model_spec):
        model_func, model_params = model_spec
        model = ClusterModelWrapper(model_func(**model_params))
        padded = -1 * np.ones(X.shape[0])
        try:
            labels = model.fit_predict(X[mask])
            padded[mask] = labels
            centers = model.model.cluster_centers_
        except(ValueError):
            padded[mask] = 0
            centers = np.mean(X[mask], axis=0).reshape((1, -1))
        return padded, centers
    
    def fit_level(self, X, parent):
        if parent.depth == len(self.config):
            return
        
        labels, centers = self.fit_model(X, parent.mask, self.config[parent.depth])
        
        for cluster in range(int(max(labels))+1):
            cluster_mask = (labels == cluster) * parent.mask
            cluster_center = centers[cluster]
            node = parent.add_child(mask=cluster_mask, center=cluster_center)
            
            self.fit_level(X, node)
            
    def fit(self, X):
        self.n_samples = X.shape[0]
        self.root = Node(depth=0, mask=np.ones(X.shape[0], dtype=bool), center=None)
        self.fit_level(X, self.root)
            
    def decode(self, docs=None, vectorizer=None):
        assignments = np.zeros((self.n_samples, len(self.config)), dtype=int)
        nodes = [node for node in self.root.children]
        centers = {}
        i = 1

        while len(nodes) > 0:
            cur = nodes.pop(0)
            assignments[cur.mask, cur.depth-1] = i
            topics = []
            if docs is not None and vectorizer is not None:
                topics = self.summarize_cluster(docs=docs[cur.mask],
                                                center=cur.center,
                                                vectorizer=vectorizer)
            centers[i] = (topics, cur.center)
            i += 1
            nodes += cur.children
            
        return assignments, centers
    
    def fit_predict(self, X, docs=None, vectorizer=None):
        self.fit(X)
        return self.decode(docs, vectorizer)
    
    @staticmethod
    def get_bigrams(docs):
        bigrams = {}
        grams = [' '.join([tex, text.split()[i+1]]) for text in docs 
                 for i, tex in enumerate(text.split()) if i < len(text.split())-1]
        for bigram in grams:
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        bigrams = pd.DataFrame.from_dict(bigrams, orient='index').reset_index().sort_values(0, ascending=False)
        bigrams.columns = ['bigram', 'count']
        bigrams = bigrams.reset_index(drop=True)
        return bigrams
    
    def summarize_cluster(self, docs, center, vectorizer):
        n_candidates = self.n_candidates
        n_return = self.n_return
        bigrams = self.get_bigrams(docs)
        possible_topics = bigrams.head(n_candidates)['bigram']
        possible_topics_vects = vectorizer.predict(possible_topics)
        sims = np.array([cosine_similarity(x, center) for x in possible_topics_vects])
        inds = np.argpartition(sims, -n_return)[-n_return:]
        inds = inds[np.argsort(sims[inds])]
        topics = possible_topics[inds].values
        return topics
