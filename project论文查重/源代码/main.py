from gensim import corpora, models, similarities
import gensim
import numpy as np
import jieba
import sys
import re


def drop_punctuation(text):
    punc = '”~`!#$%^&*()_+-=|\';"＂:/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》{《}】【\n\]\[ '
    new_text = re.sub(r"[%s]+" % punc, "", text)
    return new_text


def Separatesentence(words):
    texts = [jieba.lcut(text)for text in words]
    return texts


def main():
    arg = sys.argv  # 读取参数(文件目录)
    f = open(arg[1], 'r', encoding='utf-8')
    f2 = open(arg[2], 'r', encoding='utf-8')
    lines = f.read() #读文件
    lines_del = drop_punctuation(lines)
    lines_sep = Separatesentence([lines_del])                       # 标准文件分词数组
    lines_sep.append(['占位'])                                       #列表添加列表元素，使得lines_sep元素都是列表，将为一维
    line2 =jieba.lcut(drop_punctuation(f2.read()))                  # 查重文件分词数组
    f.close()
    f2.close()
    print("标准文件", lines_sep)
    print("标准文件维度"f'{(np.array(lines_sep)).shape}')
    print('要查重的文件', line2)
    print("line2维度"f'{(np.array(line2)).shape}')
    dictionary = corpora.Dictionary(lines_sep)                      # 唯一词典
    num_features = len(dictionary.token2id)                         # dictionary.token2id   为词语打上标签
    print('词典：', dictionary.token2id)
    corpus = [dictionary.doc2bow(text)for text in lines_sep]        # dictionary.doc2bow(text) 统计每个词语重复的次数
    print('语料库：', corpus)
    print("corpus向量", corpus)
    print("字库向量维度"f'{(np.array(corpus)).shape}')
    # corpus = dictionary.doc2bow(lines_sep)
    # new_vec = dictionary.doc2bow(line2)
    new_vec = [dictionary.doc2bow(text) for text in [line2]]
    print("查重文件new_vec向量", new_vec[0])                          #new_vec向量为三维，取首二维元素
    print("new_vec向量维度"f'{(np.array(new_vec[0])).shape}')
    # corpu = np.array(new_vec)
    # print(f'{corpu.shape}')
    print('====================================================')
    tfidf = models.TfidfModel(corpus, dictionary=dictionary)        #构建TF-IDF模型，用corpus来训练模型
    corpus_tfidf = tfidf[corpus]
    test_vec_tfidf = tfidf[new_vec[0]]
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary.keys()))
    print('\nTF-IDF模型的稀疏向量集：')
    for a in corpus_tfidf:
        print(a)
    print('\nTF-IDF模型的查重文件稀疏向量：')
    print(test_vec_tfidf)
    print('\n相似度计算：')
    sim = index[test_vec_tfidf]
    print(sim[0])
    np.savetxt(arg[3]+'\结果.txt',sim)                    #查重结果保存本地

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
