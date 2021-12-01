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
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x: vector 1
        y: vector 2
        
    Returns:
        osine similarity between x and y
    """
    return 1 - spatial.distance.cosine(x, y)


def pairwise_cosine_similarity(x):
    """
    Calculate pairwise cosine similarities between all vector pairs in x.
    
    Args:
        x: collection of vectors
        
    Returns:
        pairwise cosine similarities between all vectors in x. Length of output is len(x)C2.
    """
    return [cosine_similarity(pair[0], pair[1]) for pair in combinations(x, 2)]


class ClusterModelWrapper:
    """
    Wrapper around sklearn clustering models that provide some additional utilities, like validating
    parameters and standardizing access to the model's cluster centers. Final model stored in self.model.
    """
    
    def __init__(self, model_func, validate: dict=None, **kwargs):
        """
        Args:
            model_func: sklearn clustering model
            validate: dictionary of parameters to validate. 
                Follows the format {<param>: {'values': [<values>]}, ...}
            **kwargs: parameters to pass to model_func
        """
        self.model_func = model_func
        self.model_params = kwargs
        self.validate = validate
        self.model = None
    
    @staticmethod    
    def _cosine_measure(X, labels, lmbda=0.02):
        """
        Calculate the mean over averaged pairwise cosine similarities within all clusters. Also applies
        regularization w.r.t the number of clusters.
        
        Args:
            X: data
            labels: cluster labels
            lmbda: regularization parameter
            
        Returns:
            mean pairwise cosine similarity over all clusters
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        cosines = np.zeros(n_clusters)
        for i, label in enumerate(unique_labels):
            cluster = X[labels == label]
            cosines[i] = np.nanmean(pairwise_cosine_similarity(cluster))
        return np.nanmean(cosines) - lmbda*n_clusters
        
    def _validate(self, X, y=None):
        """
        Validate parameters in self.validate by scoring each parameter choice using the mean pairwise cosine
        similarities within clusters.
        
        Args:
            X: data
            y: true cluster labels
        
        Returns:
            dictionary of best parameters, {<param>: <best_value>, ...}
        """
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
        """
        Performs validation, then initializes and fits model_func with validated and specified parameters.
        
        Args:
            X: data
            y: true cluster labels
        """
        best_vals = self._validate(X, y)
        self.model = self.model_func(**best_vals, **self.model_params)
        return self.model.fit(X, y)
    
    def fit_predict(self, X, y=None):
        """
        Performs validation, initializes model_func with validated and specified parameters, and calls
        fit_predict.
        
        Args:
            X: data
            y: true cluster labels
            
        Returns:
            labels: predicted cluster labels
        """
        best_vals = self._validate(X, y)
        self.model = self.model_func(**best_vals, **self.model_params)
        return self.model.fit_predict(X, y)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)
        
    def get_n_clusters(self):
        """
        Get the number of clusters in the model, which can be stored differently for different sklearn
        clustering models.
        
        Returns:
            n_clusters: number of clusters
        """
        try:
            return self.model.n_clusters
        except(AttributeError):
            return self.model.cluster_centers_.shape[0]            
        

class Node:
    """
    Node for representing hierarchical topic clustering model as a tree. A node represents a cluster.
    A node can have multiple children, which represent sub-clusters beneath it.
    """
    
    def __init__(self, depth, mask, center):
        """
        Args:
            depth: depth of node in tree, indicating what level it is in the hierarchy
            mask: boolean mask indicating which rows in X belong to this cluster
            center: cluster center representation
        """
        self.depth = depth
        self.mask = mask
        self.center = center
        self.children = []
        
    def add_child(self, mask, center):
        """
        Add a child node to this node representing a sub-cluster beneath it.
        
        Args:
            mask: boolean mask indicating which rows in X belong to the sub-cluster
            center: sub-cluster center representation
            
        Returns:
            child: child node
        """
        child = Node(self.depth+1, mask, center)
        self.children.append(child)
        return child
        
        
class TopicTreeModel:
    """
    Hierarchical topic clustering model built from multiple sklearn flat clustering models. Every level in the
    hierarchy is associated with one clustering model.
    Represented as a tree, where each node represents a cluster.
    """
    
    def __init__(self, config, n_candidates=10, n_return=3):
        """
        Args:
            config: list of clustering model functions and their parameters. Passed to ClusterModelWrapper,
            so validate is supported, which is particularly useful for models that need n_clusters to be specified. 
                Follows the format: [(<model_func>, {<param>: <values>}), ...]
                Ex: [(cluster.KMeans, {'n_clusters': 2}), 
                     (cluster.AgglomerativeClustering, {'validate': {'linkage': {'values': ['ward', 'complete', 'average']}}})]
            n_candidates (optional): Number of bigrams to consider when decoding the tree and summarizing clusters
                into topic strings. Default is 10.
            n_return (optional): Number of candidate topic strings to return for each cluster when decoding. 
                Default is 3.
        """
        self.config = config
        self.root = None
        self.n_candidates = n_candidates
        self.n_return = n_return
        
    @staticmethod   
    def fit_model(X, mask, model_spec):
        """
        Fit a clustering model according to model_spec to the data indicated by the mask.
        
        Args:
            X: data
            mask: boolean mask indicating which rows in X belong to the cluster that should be sub-clustered
            model_spec: clustering model function and its parameters, to pass to ClusterModelWrapper
            
        Returns:
            padded_labels: cluster labels for entire X. Rows that are not in the mask are labeled as -1.
            centers: cluster centers for each cluster.
        """
        model_func, model_params = model_spec
        model = ClusterModelWrapper(model_func, **model_params)
        padded_labels = -1 * np.ones(X.shape[0])
        try:
            labels = model.fit_predict(X[mask])
            padded_labels[mask] = labels
            centers = model.model.cluster_centers_
        except(ValueError, AttributeError):
            padded_labels[mask] = 0
            centers = np.mean(X[mask], axis=0).reshape((1, -1))
        return padded_labels, centers
    
    def fit_level(self, X, parent):
        """
        Fit a clustering model to every cluster in the current hierarchy level, and update tree.
        
        Ex. If the current level is the root, fit one clustering model - the first model in self.config - 
        to all data in X. If the current level is the second level, fit a clustering model (specified in 
        self.config) to every cluster generated in the first level, only the data that belongs to that cluster. 
        
        Args:
            X: data
            parent: parent node in tree, representing parent cluster
        """
        if parent.depth == len(self.config):
            return
        
        labels, centers = self.fit_model(X, parent.mask, self.config[parent.depth])
        
        for cluster in range(int(max(labels))+1):
            cluster_mask = (labels == cluster) * parent.mask
            cluster_center = centers[cluster]
            node = parent.add_child(mask=cluster_mask, center=cluster_center)
            
            self.fit_level(X, node)
            
    def fit(self, X, y=None):
        """
        Fit the hierarchical topic clustering model to the data.
        
        Args:
            X: data
            y (optional): Ignored, kept for API compatibility.
        """
        self.n_samples = X.shape[0]
        self.root = Node(depth=0, mask=np.ones(X.shape[0], dtype=bool), center=None)
        self.fit_level(X, self.root)
            
    def decode(self, docs=None, vectorizer=None):
        """
        Decode the hierarchical topic model tree to summarize every cluster with its center representation and
        mask, denoting which rows in X belong to that cluster.
        
        Can optionally provide the actual document strings corresponding to the training data, and a text vectorizer. 
        In this case a topic string will be generated for each cluster.
        
        Args:
            docs (optional): documents corresponding to training data X. If specified, vectorizer should also 
                be specified.
            vectorizer (optional): vectorizer to use to evaluate topic strings. If specified, docs should also 
                be specified.
                
        Returns:
            assignments: list of cluster assignments for each row in training data X
            centers: dict of cluster summaries composed of 
                {'topics': <topic strings>, 'center': <center representation>, 'mask': <mask>}
        """
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
        """
        Fit model to X and decode the tree to summarize clusters (see TopicTreeModel.decode).
        
        Args:
            X: data
            y (optional): document strings corresponding to training data X. If specified, vectorizer should also
                be specified.
            vectorizer (optional): vectorizer to use to evaluate topic strings. If specified, docs should also
                be specified.
                
        Returns:
            assignments: list of cluster assignments for each row in training data X
            centers: dict of cluster summaries composed of 
                {'topics': <topic strings>, 'center': <center representation>, 'mask': <mask>}
        """
        self.fit(X)
        return self.decode(y, vectorizer)
    
    @staticmethod
    def get_bigrams(docs):
        """
        Get all bigrams in docs as well as their counts. If a doc is only one token, the unigram is added.
        
        Args:
            docs: collection of documents
            
        Returns:
            pd.DataFrame of bigrams and their counts
        """
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
        """
        Generate topic strings for a cluster, by collecting the top self.n_candidates bigrams from documents
        in the cluster, vectorizing them with the vectorizer, then taking the top self.n_topics topics based
        on their cosine similarity to the cluster center. 
        
        Args:
            docs: documents in cluster
            center: cluster center
            vectorizer: vectorizer to use to vectorize bigrams
            
        Returns:
            topics: self.n_topics topic strings
        """
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
        """
        Create pd.DataFrame of data assignments and topic hierarchy in topic string form, one column per level.
        
        Args:
            docs: documents corresponding to training data X
            assignments: list of cluster assignments for each row in training data X
            clusters: cluster summaries composed of 
                {'topics': <topic strings>, 'center': <center representation>, 'mask': <mask>}
        
        Returns:
            pd.DataFrame of shape (len(docs), len(config)+1)
        """
        output = pd.DataFrame(docs)
        for level in range(assignments.shape[1]):
            output[f'level{level+1}'] = pd.Series(map(lambda x: clusters[x]['topics'], assignments[:, level]), 
                                                name=f'level{level+1}')
        return output
