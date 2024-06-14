# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/6/14 9:43
# @Function: 本脚本用于比较和识别给定数据集中基于其处理过的词集的相似问题。

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 文件路径常量
WORDS_SET_PATH = '..\\dat\\words_set\\words_set.csv'
TESTDATA_PATH = '..\\dat\\raw_data\\testdata.csv'


def load_data(filepath):
    """从CSV文件加载数据，确保正确的编码和清洁的列名。

    参数:
        filepath (str): 文件路径。

    返回:
        DataFrame: 载入的数据。
    """
    data = pd.read_csv(filepath)
    data.columns = [col.strip().replace('"', '') for col in data.columns]
    return data


def vectorize_data(data):
    """使用TF-IDF向量化处理'Processed Words Set'。

    参数:
        data (DataFrame): 包含需要向量化文本的DataFrame。

    返回:
        tuple: 向量化后的矩阵和使用的向量器。
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data['Processed Words Set'].tolist()), vectorizer


def find_high_similarity_pairs(cosine_sim_matrix, data, threshold=0.6):
    """识别高余弦相似度的问题对。

    参数:
        cosine_sim_matrix (array): 余弦相似度矩阵。
        data (DataFrame): 包含问题的DataFrame。
        threshold (float): 相似度阈值。

    返回:
        list: 高相似度问题对列表。
    """
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if cosine_sim_matrix[i, j] > threshold:
                pairs.append((data.iloc[i]['Question'], data.iloc[j]['Question']))
    return pairs


def main():
    logging.info("Loading datasets.")
    words_data = load_data(WORDS_SET_PATH)
    test_data = load_data(TESTDATA_PATH)

    logging.info("Vectorizing data.")
    X, vectorizer = vectorize_data(words_data)

    logging.info("Calculating cosine similarity.")
    cosine_sim = cosine_similarity(X)

    logging.info("Finding high similarity pairs.")
    high_similarity_pairs = find_high_similarity_pairs(cosine_sim, words_data, 0.6)

    logging.info(f"Total high similarity pairs found: {len(high_similarity_pairs)}")

    for pair in high_similarity_pairs:
        try:
            matched_ids_1 = test_data[test_data['问题描述'].str.contains(pair[0], regex=False, na=False)][
                '问题ID'].tolist()
            matched_ids_2 = test_data[test_data['问题描述'].str.contains(pair[1], regex=False, na=False)][
                '问题ID'].tolist()
            if matched_ids_1 and matched_ids_2:
                logging.info(
                    f"Found question IDs: '{matched_ids_1[0]}': '{pair[0]}', '{matched_ids_2[0]}': '{pair[1]}'")
            else:
                if not matched_ids_1:
                    logging.warning(f"No match found for: {pair[0]}")
                if not matched_ids_2:
                    logging.warning(f"No match found for: {pair[1]}")
        except KeyError as e:
            logging.error(f"Column name may be incorrect, check column name: {str(e)}")


if __name__ == '__main__':
    main()
