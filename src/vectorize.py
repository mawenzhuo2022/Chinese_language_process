# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/5/22 10:51
# @Function: 本脚本使用TF-IDF向量化和余弦相似度计算，根据给定的已处理输入找出数据集中最相似的问题。

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志，为调试提供详细信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """从CSV文件加载数据。
    参数:
        file_path (str): 包含数据的CSV文件路径。
    返回:
        DataFrame: 包含加载数据的pandas DataFrame。
    异常:
        Exception: 如果文件加载失败，记录错误并退出程序。
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"从 {file_path} 加载数据失败: {e}")
        raise  # 这将错误传递给调用者，可以决定如何处理


def vectorize_data(data):
    """使用TF-IDF向量化文本数据。
    参数:
        data (DataFrame): 包含文本数据列'Processed Words Set'的pandas DataFrame。
    返回:
        tuple: 包含向量器(TfidfVectorizer对象)和TF-IDF矩阵的元组。
    异常:
        Exception: 如果向量化失败，记录错误并退出程序。
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Processed Words Set'])
        return vectorizer, tfidf_matrix
    except Exception as e:
        logging.error(f"数据向量化失败: {e}")
        raise


def find_most_similar_questions(new_input, vectorizer, tfidf_matrix, data, top_n=5):
    """基于TF-IDF向量的余弦相似度找出最相似的问题。
    参数:
        new_input (str): 要比较的已处理输入文本。
        vectorizer (TfidfVectorizer): 用于转换文本数据的向量器。
        tfidf_matrix (array): 从数据集向量化得到的TF-IDF矩阵。
        data (DataFrame): 原始数据集的DataFrame。
        top_n (int): 返回的最相似问题的数量。
    返回:
        list: 包含最相似问题及其相似度得分的列表。
    异常:
        Exception: 如果计算相似度失败，记录错误。
    """
    try:
        new_input_tfidf = vectorizer.transform([new_input])
        cosine_similarities = cosine_similarity(new_input_tfidf, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        return [(data.iloc[index]['Question'], cosine_similarities[index]) for index in top_indices]
    except Exception as e:
        logging.error(f"查找相似问题时出错: {e}")
        raise


def main(processed_input):
    """主函数，加载数据，向量化并找出相似问题。
    参数:
        processed_input (str): 需要找出相似问题的已处理文本输入。
    """
    file_path = '../dat/words_set/words_set.csv'
    data = load_data(file_path)
    vectorizer, tfidf_matrix = vectorize_data(data)
    similar_questions = find_most_similar_questions(processed_input, vectorizer, tfidf_matrix, data)
    for question, similarity in similar_questions:
        logging.info(f"{question}: 相似度 = {similarity}")


# 使得此模块既可以命令行调用也可以作为库函数调用
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = sys.argv[1]  # 命令行参数传入已处理文本
    else:
        input_text = input("请输入已处理的文本: ")  # 交互式输入已处理文本
    main(input_text)
