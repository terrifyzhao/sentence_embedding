from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
from bert_embedding import SentenceEmbedding

# model = Word2Vec()
# model = SentenceEmbedding('/Users/joezhao/Documents/pretrain model/chinese_bert_L-12_H-768_A-12', mode='two_avg')


model = SentenceEmbedding('/home/joska/ptm/bert', mode='two_avg')


def cal_cosine(a, b):
    return np.sum(np.multiply(a, b), axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def cal_pearson():
    df = pd.read_csv('data/LCQMC.csv')
    label = df['label'].values

    vec1 = model.encode(df['sentence1'].values.tolist())
    vec2 = model.encode(df['sentence2'].values.tolist())
    res = cal_cosine(vec1, vec2)

    print('pearson:', pearsonr(res, label)[0])
    print('spearman:', spearmanr(res, label)[0])


if __name__ == '__main__':
    cal_pearson()
