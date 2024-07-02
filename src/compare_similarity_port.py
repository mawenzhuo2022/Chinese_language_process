# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/6/14 15:34
# @Function: 文本相似度检测脚本，用于检测输入的文本与数据库中文本的相似度
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义文件路径
WORDS_SET_PATH = '../dat/words_set/words_set.csv'
STOP_WORDS_PATH = '../dat/stop_words/stop_words_full.txt'

class TextPreprocessor:
    """文本预处理类，负责文本的停用词去除和特殊字符清理"""

    def __init__(self, stop_words_file):
        """构造函数：加载停用词文件"""
        self.stop_words = self.load_stop_words(stop_words_file)

    def load_stop_words(self, file_path):
        """从指定路径加载停用词集"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stop_words = {line.strip() for line in file}
                logging.info("已加载停用词数量：{}".format(len(stop_words)))
                return stop_words
        except FileNotFoundError:
            logging.error("找不到停用词文件：{}".format(file_path))
            return set()

    def preprocess(self, text):
        """对输入的文本进行预处理，包括去除数字和特殊字符，使用结巴分词进行中文分词，并去除停用词"""
        text = re.sub(r'[^\w\s]', ' ', text)  # 去除特殊字符
        text = re.sub(r'\d+', '', text)  # 去除数字
        return ' '.join([word for word in jieba.cut(text) if word not in self.stop_words and not word.isspace()])

def load_data(filepath):
    """从指定路径加载数据集"""
    data = pd.read_csv(filepath)
    data.columns = [col.strip().replace('"', '') for col in data.columns]
    return data

def vectorize_data(data, input_text, preprocessor):
    """向量化处理输入文本与数据集，以便进行相似度检测"""
    processed_text = preprocessor.preprocess(input_text)
    vectorizer = TfidfVectorizer()
    all_text = data['Processed Words Set'].tolist() + [processed_text]
    X = vectorizer.fit_transform(all_text)
    return X[:-1], X[-1], vectorizer

def check_similarity(input_vector, dataset_vector, threshold=0.6):
    """检查输入文本与数据集中文本的相似度，返回相似度结果"""
    cosine_sim = cosine_similarity(input_vector, dataset_vector)
    max_similarity = cosine_sim.max()
    if max_similarity > threshold:
        return {"error": "输入的文本与现有条目过于相似", "similarity_score": max_similarity}
    else:
        return {"message": "输入的文本具有足够的区别度", "similarity_score": max_similarity}

@app.route('/similarity_check', methods=['POST'])
def similarity_check():
    """处理POST请求，检查文本相似度"""
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({"error": "未提供文本"}), 400
    try:
        preprocessor = TextPreprocessor(STOP_WORDS_PATH)
        words_data = load_data(WORDS_SET_PATH)
        dataset_vector, input_vector, vectorizer = vectorize_data(words_data, input_text, preprocessor)
        result = check_similarity(input_vector, dataset_vector)
        return jsonify(result), 200
    except Exception as e:
        logging.error("处理异常：" + str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
