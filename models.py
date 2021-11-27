from itertools import combinations
import warnings
import logging

import numpy as np
import pandas as pd
from scipy import spatial

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(level = logging.INFO)
warnings.filterwarnings('ignore')


def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)


def pairwise_cosine_similarity(x):
    return [cosine_similarity(pair[0], pair[1]) for pair in combinations(x, 2)]


class ClusterModelWrapper:
    def __init__(self, model_func, validate: dict=None, **kwargs):
        self.model_func = model_func
        self.model_params = kwargs
        self.validate = validate
        self.model = None
    
    @staticmethod    
    def _cosine_measure(X, labels, lmbda=0.02):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        cosines = np.zeros(n_clusters)
        for i, label in enumerate(unique_labels):
            cluster = X[labels == label]
            cosines[i] = np.nanmean(pairwise_cosine_similarity(cluster))
        return np.nanmean(cosines) - lmbda*n_clusters
        
    def _validate(self, X, y=None):
        best_vals = {}
        if self.validate:
            all_scores = {}
            for param, to_val in self.validate.items():
                all_scores[param] = {}
                best_score, best_v = 0, 0
                for v in to_val['values']:
                    model = self.model_func(**{param: v}, **self.model_params)
                    labels = model.fit_predict(X, y)
                    score = self._cosine_measure(X, labels, lmbda=to_val.get('lambda', 0))
                    all_scores[param][v] = score
                    if score > best_score:
                        best_score = score
                        best_v = v
                best_vals[param] = best_v
        return best_vals
    
    def fit(self, X, y=None):
        best_vals = self._validate(X, y)
        self.model = self.model_func(**best_vals, **self.model_params)
        return self.model.fit(X, y)
    
    def fit_predict(self, X, y=None):
        best_vals = self._validate(X, y)
        self.model = self.model_func(**best_vals, **self.model_params)
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
        model = ClusterModelWrapper(model_func, **model_params)
        padded = -1 * np.ones(X.shape[0])
        try:
            labels = model.fit_predict(X[mask])
            padded[mask] = labels
            centers = model.model.cluster_centers_
        except(ValueError, AttributeError):
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
            
    def fit(self, X, y=None):
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
            centers[i] = {'topics': topics, 'center': cur.center, 'mask': cur.mask}
            i += 1
            nodes += cur.children
            
        return assignments, centers
    
    def fit_predict(self, X, y=None, vectorizer=None):
        self.fit(X)
        return self.decode(y, vectorizer)
    
    @staticmethod
    def get_bigrams(docs):
        bigrams = {}
        grams = [' '.join([tex, text.split()[i+1]]) for text in docs 
                 for i, tex in enumerate(text.split()) if i < len(text.split())-1]
        if len(grams) == 0:
            # use unigrams
            grams = [text for text in docs]
        for bigram in grams:
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        bigrams = pd.DataFrame.from_dict(bigrams, orient='index').reset_index().sort_values(0, ascending=False)
        bigrams.columns = ['bigram', 'count']
        bigrams = bigrams.reset_index(drop=True)
        return bigrams
    
    def summarize_cluster(self, docs, center, vectorizer):
        n_candidates = self.n_candidates
        bigrams = self.get_bigrams(docs)
        possible_topics = bigrams.head(n_candidates)['bigram']
        possible_topics_vects = vectorizer.predict(possible_topics)
        n_return = min(self.n_return, len(possible_topics))
        sims = np.array([cosine_similarity(x, center) for x in possible_topics_vects])
        inds = np.argpartition(sims, -n_return)[-n_return:]
        inds = inds[np.argsort(sims[inds])]
        topics = possible_topics[inds].values
        return topics

    @staticmethod
    def flatten(docs, assignments, clusters):
        output = pd.DataFrame(docs)
        for level in range(assignments.shape[1]):
            output[f'level{level+1}'] = pd.Series(map(lambda x: clusters[x]['topics'], assignments[:, level]), 
                                                name=f'level{level+1}')
        return output
