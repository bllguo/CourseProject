import numpy as np

from abc import ABC, abstractmethod


def tokenize(s):
    return s.split()


class TextVectorizer(ABC):
    """
    Base class for creating vector representations of text.
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or tokenize
    
    @abstractmethod
    def fit(self, X, y=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class CountVectorizer(TextVectorizer):
    def __init__(self, unk=False, threshold=0, **kwargs):
        super().__init__(**kwargs)
        self.unk = unk
        self.vocabulary = {'UNK': 0} if unk else {}
        self.threshold = threshold
        self.term_freqs = {}
        
    def build_vocabulary(self, documents):
        term_freqs = {}
        for doc in documents:
            tokens = self.tokenizer(doc)
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
        self.term_freqs = term_freqs
        
        i = 1 if self.unk else 0
        for term, count in self.term_freqs.items():
            if count >= self.threshold:
                self.vocabulary[term] = i
                i += 1
               
    def fit(self, X, y=None):
        self.build_vocabulary(X)
                    
    def predict(self, documents):
        v = len(self.vocabulary)
        n = len(documents)
        out = np.zeros((n, v))
        for i, doc in enumerate(documents):
            tokens = self.tokenizer(doc)
            for token in tokens:
                j = self.vocabulary.get(token, 0 if self.unk else None)
                if j is not None:
                    out[i, j] += 1
        return out
            
    
class TfidfVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.document_frequencies = {}
                    
    def fit(self, X, y=None):
        self.build_vocabulary(X)
        
        for doc in X:
            tokens = self.tokenizer(doc)
            for token in tokens:
                j = self.vocabulary.get(token)
                if j is not None:
                    self.document_frequencies[j] = self.document_frequencies.get(j, 0) + 1
                    
    def predict(self, X):
        v = len(self.vocabulary)
        n = len(X)
        out = np.zeros((n, v))
        for i, doc in enumerate(X):
            tokens = self.tokenizer(doc)
            d = len(tokens)
            term_counts = {}
            for token in tokens:
                j = self.vocabulary.get(token, 0 if self.unk else None)
                if j is not None:
                    term_counts[j] = term_counts.get(j, 0) + 1
            for j, cnt in term_counts.items():
                tf = cnt/d
                idf = np.log(n/(self.document_frequencies.get(j, 0) + 1))
                out[i, j] = tf*idf
        return out
                
    
class EmbeddingVectorizer(TextVectorizer):
    def __init__(self, embeddings: dict=None, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self.embedding_size = embeddings[list(embeddings.keys())[0]].shape[0]
        
    def fit(self, X=None, y=None):
        pass
    
    def predict(self, X):
        n = len(X)
        out = np.zeros((n, self.embedding_size))
        for i, doc in enumerate(X):
            v = np.zeros(self.embedding_size)
            count = 0
            tokens = self.tokenizer(doc)
            for token in tokens:
                word_embedding = self.embeddings.get(
                    token, 
                    self.embeddings.get(
                        token.capitalize(), 
                        self.embeddings.get(token.lower())))
                if word_embedding is not None:
                    v += word_embedding
                    count += 1
            if count > 0:
                v /= count
            out[i, :] = v
        return out
