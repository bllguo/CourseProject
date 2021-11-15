import numpy as np

from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize


class TextVectorizer(ABC):
    """
    Base class for creating vector representations of text.
    """
    @abstractmethod
    def fit(self, documents):
        pass
    
    @abstractmethod
    def predict(self, documents):
        pass


class CountVectorizer(TextVectorizer):
    def __init__(self, unk=False):
        self.unk = unk
        self.vocabulary = {'UNK': 0} if unk else {}
               
    def fit(self, documents):
        i = 1 if self.unk else 0
        for doc in documents:
            tokens = word_tokenize(doc)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = i
                    
    def predict(self, documents):
        v = len(self.vocabulary)
        n = len(documents)
        out = np.zeros((n, v))
        for i, doc in enumerate(documents):
            tokens = word_tokenize(doc)
            for token in tokens:
                j = self.vocabulary.get(token, 0 if self.unk else None)
                if j:
                    out[i, j] += 1
        return out
            
    
class TfidfVectorizer(CountVectorizer):
    def __init__(self, unk=False):
        self.unk = unk
        self.vocabulary = {'UNK': 0} if unk else {}
        self.document_frequencies = {}
                    
    def fit(self, documents):
        i = 1 if self.unk else 0
        for doc in documents:
            tokens = word_tokenize(doc)
            indices = set()
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = i
                indices.add(self.vocabulary[token])
            for j in indices:
                self.document_frequencies[j] = self.document_frequencies.get(j, 0) + 1
                    
    def predict(self, documents):
        v = len(self.vocabulary)
        n = len(documents)
        out = np.zeros((n, v))
        for i, doc in enumerate(documents):
            tokens = word_tokenize(doc)
            d = len(tokens)
            term_counts = {}
            for token in tokens:
                j = self.vocabulary.get(token, 0 if self.unk else None)
                if j:
                    term_counts[j] = term_counts.get(j, 0) + 1
            for j, cnt in term_counts.items():
                tf = cnt/d
                idf = np.log(n/(self.document_frequencies.get(j, 0) + 1))
                out[i, j] = tf*idf
        return out
                
    
class EmbeddingVectorizer(TextVectorizer):
    def __init__(self, embeddings: dict=None):
        self.embeddings = embeddings
        self.embedding_size = embeddings[list(embeddings.keys())[0]].shape[0]
        
    def fit(self, documents):
        pass
    
    def predict(self, documents):
        n = len(documents)
        out = np.zeros((n, self.embedding_size))
        for i, doc in enumerate(documents):
            v = np.zeros(self.embedding_size)
            count = 0
            tokens = word_tokenize(doc)
            for token in tokens:
                word_embedding = self.embeddings.get(token, 0)
                v += word_embedding
                count += 1 if word_embedding != 0 else 0
            v /= count
            out[i, :] = v
