from gensim.models.word2vec import Word2VecKeyedVectors
import jieba
import numpy as np
from tqdm import tqdm


class SentenceEmbedding:
    def __init__(self, path):
        self.model = Word2VecKeyedVectors.load_word2vec_format(path, binary=False)

    def encode(self, sentences):
        embeddings = []
        for sentence in tqdm(sentences):
            segment = list(jieba.cut(sentence))
            embedding = np.zeros(shape=(200,))
            for s in segment:
                try:
                    embedding += self.model[s]
                except:
                    embedding += np.array([1e-8] * 200)
            embeddings.append(embedding)
        return np.array(embeddings)
