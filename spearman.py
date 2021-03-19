from scipy.stats import spearmanr, pearsonr
import pandas as pd
from word2vec.word2vec import Word2Vec
import numpy as np
from tqdm import tqdm
from sentence_embedding import SentenceEmbedding

# model = Word2Vec()
# model = SentenceEmbedding('/Users/joezhao/Documents/pretrain model/chinese_bert_L-12_H-768_A-12')
model = SentenceEmbedding('/home/joska/ptm/bert')


def cal_cosine(a, b):
    return np.sum(np.multiply(a, b), axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def cal_pearson():
    df = pd.read_csv('data/LCQMC.csv')
    label = df['label'].values

    vec1 = []
    for s in tqdm(df['sentence1']):
        vec1.append(model.encode(s))

    vec2 = []
    for s in tqdm(df['sentence2']):
        vec2.append(model.encode(s))

    res = cal_cosine(np.array(vec1), np.array(vec2))

    print('pearson:', pearsonr(res, label)[0])
    print('spearman:', spearmanr(res, label)[0])


if __name__ == '__main__':
    cal_pearson()
