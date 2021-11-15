from datetime import datetime
import logging
import os
import tempfile
import gensim.downloader as api
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import Word2Vec, KeyedVectors, callbacks
from gensim.test.utils import get_tmpfile

logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.INFO)


class StreamCorpus:
    """
    Gensim-compatible corpus class that streams a text file instead of storing it all in memory,
    allowing for much more efficient training.
    """
    def __init__(self, filename: str='corpus.txt', dct: Dictionary=None):
        self.dct = dct
        self.filename = filename
        
    def __iter__(self):
        with open(self.filename, 'r', encoding="utf8") as f:
            for line in f:
                out = line.lower().split()
                if self.dct:
                    out = self.dct.doc2bow(line.lower().split())
                yield out


class Word2VecTuner():
    """
    Wrapper around gensim.models.Word2Vec that supports transfer learning.
    Allows loading pretrained embeddings into the Word2Vec model before doing additional
    training to fine-tune the embeddings.
    """
    def __init__(self, **kwargs):
        self.model = Word2Vec(**kwargs)
        self.model.build_vocab(list(self.embeddings.index_to_key))
        
    def load_embeddings(self, embeddings: KeyedVectors=None, gensim_model: str=None):
        if gensim_model:
            logging.info(f'Downloading {gensim_model}...')
            embeddings = api.load(gensim_model)
            logging.info(f'Download complete.')
        logging.info('Updating model vocabulary...')    
        self.model.build_vocab(list(embeddings.index_to_key))
        logging.info('Update complete.')
        logging.info('Loading pretrained embeddings...')
        with tempfile.TemporaryDirectory() as temp_path:
            fpath = os.path.join(temp_path, 'embeddings.txt')
            embeddings.save_word2vec_format(fpath, binary=False)
            self.model.wv.vectors_lockf = np.ones(len(self.model.wv), dtype=np.float32)
            self.model.wv.intersect_word2vec_format(fpath, lockf=1.0)
        logging.info('Pretrained embeddings loaded to Word2Vec model.')
        
    def train(self, corpus, **kwargs):
        self.model.build_vocab(corpus, update=True)
        self.model.train(corpus, **kwargs)


class EpochSaver(callbacks.CallbackAny2Vec):
    """
    Handy callback to save model after each training epoch.
    """
    def __init__(self, path_prefix: str):
        self.path_prefix = path_prefix
        self.epoch = 0
        self.epoch_start = None

    def on_epoch_begin(self, model):
        self.epoch_start = datetime.now()
        print(f"Epoch #{self.epoch} start: {self.epoch_start}")
        
    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        print(f"Epoch #{self.epoch} end: {datetime.now()}")
        print(f"Elapsed: {datetime.now() - self.epoch_start}")
        self.epoch += 1
