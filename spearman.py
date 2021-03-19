from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
# from word2vec import SentenceEmbedding
# from bert_embedding import SentenceEmbedding
# from simbert_embedding import SentenceEmbedding
from sbert_embedding import SentenceEmbedding

# model = SentenceEmbedding('word2vec/word_embedding.txt')
# model = SentenceEmbedding('/Users/joezhao/Documents/pretrain model/distiluse-base-multilingual-cased-v1')
# model = SentenceEmbedding('/home/joska/ptm/distiluse-base-multilingual-cased-v1')

model = SentenceEmbedding('distiluse-base-multilingual-cased-v1')


def cal_cosine(a, b):
    return np.sum(np.multiply(a, b), axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def cal_pearson():
    df = pd.read_csv('data/LCQMC.csv')[0:10]
    label = df['label'].values

    pool_out1 = model.encode(df['sentence1'].values.tolist())
    pool_out2 = model.encode(df['sentence2'].values.tolist())

    # pool_out1, mean_out1, two_layer_out1 = model.encode(df['sentence1'].values.tolist())
    # pool_out2, mean_out2, two_layer_out2 = model.encode(df['sentence2'].values.tolist())

    res = cal_cosine(pool_out1, pool_out2)
    print('cls')
    print('pearson:', pearsonr(res, label)[0])
    print('spearman:', spearmanr(res, label)[0])

    # res = cal_cosine(mean_out1, mean_out2)
    # print('last layer avg')
    # print('pearson:', pearsonr(res, label)[0])
    # print('spearman:', spearmanr(res, label)[0])
    #
    # res = cal_cosine(two_layer_out1, two_layer_out2)
    # print('last two layers avg')
    # print('pearson:', pearsonr(res, label)[0])
    # print('spearman:', spearmanr(res, label)[0])


if __name__ == '__main__':
    cal_pearson()
