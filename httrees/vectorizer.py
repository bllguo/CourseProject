import numpy as np

from abc import ABC, abstractmethod
from gensim.models import KeyedVectors

def tokenize(s):
    return s.split()


class TextVectorizer(ABC):
    """
    Base class for creating vector representations of text, following the sklearn API.
    """
    
    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer (optional): function that takes a string and returns a list of tokens. Defaults to 
            str.split() if None specified.
        """
        self.tokenizer = tokenizer or tokenize
    
    @abstractmethod
    def fit(self, X, y=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass


class CountVectorizer(TextVectorizer):
    """
    Count-based vectorizer for text, following the sklearn API.
    """
    
    def __init__(self, unk=False, threshold=0, **kwargs):
        """
        Args:
            unk (optional): whether to include an "unknown" token. Defaults to False.
            threshold (optional): minimum frequency of a token to be included in the vocabulary. Defaults to 0.
            **kwargs: keyword arguments passed to TextVectorizer.
        """
        super().__init__(**kwargs)
        self.unk = unk
        self.vocabulary = {'UNK': 0} if unk else {}
        self.threshold = threshold
        self.term_freqs = {}
        
    def build_vocabulary(self, documents):
        """
        Builds the vocabulary from the given documents. If self.unk, then UNK is added to the vocabulary,
        and terms with frequency less than self.threshold are counted as UNK.
        
        Args:
            documents: iterable collection of documents.
        """
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
        """
        Fit vectorizer to given data.
        
        Args:
            X: iterable collection of documents.
            y (optional): Ignored, kept for API compatibility.
        """
        self.build_vocabulary(X)
                    
    def predict(self, X):
        """
        Produce vector representations of given documents. If self.unk, unknown tokens are represented
        with the corresponding UNK count.
        
        Args:
            documents: iterable collection of documents.
            
        Returns:
            numpy array of shape (len(X), len(self.vocabulary))
        """
        v = len(self.vocabulary)
        n = len(X)
        out = np.zeros((n, v))
        for i, doc in enumerate(X):
            tokens = self.tokenizer(doc)
            for token in tokens:
                j = self.vocabulary.get(token, 0 if self.unk else None)
                if j is not None:
                    out[i, j] += 1
        return out
            
    
class TfidfVectorizer(CountVectorizer):
    """
    Tf-idf based vectorizer for text, following the sklearn API.
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments passed to CountVectorizer.
        """
        super().__init__(**kwargs)
        self.document_frequencies = {}
                    
    def fit(self, X, y=None):
        """
        Fit vectorizer to given data.
        
        Args:
            X: iterable collection of documents.
            y (optional): Ignored, kept for API compatibility.
        """
        self.build_vocabulary(X)
        
        for doc in X:
            tokens = self.tokenizer(doc)
            for token in tokens:
                j = self.vocabulary.get(token)
                if j is not None:
                    self.document_frequencies[j] = self.document_frequencies.get(j, 0) + 1
                    
    def predict(self, X):
        """
        Produce vector representations of given documents. If self.unk, unknown tokens are represented
        with the corresponding UNK tf-idf value.
        
        Args:
            documents: iterable collection of documents.
            
        Returns:
            numpy array of shape (len(X), len(self.vocabulary))
        """
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
    """
    Embedding-based vectorizer for text, following the sklearn API.
    """
    
    def __init__(self, embeddings: dict, **kwargs):
        """
        Args:
            embeddings: dictionary of embeddings, where the keys are the tokens and the values are the vectors.
            **kwargs: keyword arguments passed to TextVectorizer.
        """
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self.embedding_size = embeddings[list(embeddings.keys())[0]].shape[0]
        
    def fit(self, X=None, y=None):
        """
        Fit vectorizer to given data. Does not do anything here, as embeddings are already trained.
        
        Args:
            X (optional): ignored, kept for API compatibility.
            y (optional): Ignored, kept for API compatibility.
        """
        pass
    
    def predict(self, X):
        """
        Produce vector representations of given documents.
        Unknown tokens are first checked for lowercased and capitalized versions. If not found, 
        the embedding defaults to zero.
        
        Args:
            documents: iterable collection of documents.
            
        Returns:
            numpy array of shape (len(X), self.embedding_size)
        """
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


class KVVectorizer(TextVectorizer):
    """
    Embedding-based vectorizer for text, following the sklearn API. Similar to EmbeddingVectorizer, but
    uses gensim.models.KeyedVectors instead of a dictionary, so there are no failsafes for unknown tokens.
    """
    def __init__(self, embeddings: KeyedVectors, **kwargs):
        """
        Args:
            embeddings: gensim KeyedVectors object.
        """
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self.embedding_size = embeddings.vector_size
    
    def fit(self, X=None, y=None):
        """
        Fit vectorizer to given data. Does not do anything here, as embeddings are already trained.
        
        Args:
            X (optional): ignored, kept for API compatibility.
            y (optional): Ignored, kept for API compatibility.
        """
        pass
    
    def predict(self, X):
        """
        Produce vector representations of given documents. Handling of unknown tokens is done by
        the embeddings.
        
        Args:
            documents: iterable collection of documents.
            
        Returns:
            numpy array of shape (len(X), self.embedding_size)
        """
        n = len(X)
        out = np.zeros((n, self.embedding_size))
        for i, doc in enumerate(X):
            v = np.zeros(self.embedding_size)
            count = 0
            tokens = self.tokenizer(doc)
            for token in tokens:
                word_embedding = self.embeddings[token]
                if word_embedding is not None:
                    v += word_embedding
                    count += 1
            if count > 0:
                v /= count
            out[i, :] = v
        return out
