# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/7/2 13:19
# @Function: 实现中文文本预处理和IP地址提取的Flask服务


from flask import Flask, request, jsonify
import jieba
import re
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Set, Tuple

app = Flask(__name__)

# 配置日志记录，包括时间、日志级别和日志信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InputProcessor:
    def __init__(self, stop_words_file: str = '../dat/stop_words/stop_words_full.txt', use_tfidf: bool = True,
                 ngram_range: Tuple[int, int] = (1, 1), keyword_threshold: float = 0.1):
        # 初始化处理器，加载停用词
        self.stop_words = self.load_stop_words(stop_words_file)
        # 根据是否使用TF-IDF选择不同的向量化方法
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=self.stop_words,
                                          ngram_range=ngram_range) if use_tfidf else CountVectorizer(
            tokenizer=jieba.cut, stop_words=self.stop_words, ngram_range=ngram_range)
        self.keyword_threshold = keyword_threshold

    def load_stop_words(self, file_path: str) -> Set[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stop_words = {line.strip() for line in file}
                logging.info("Stop words loaded successfully.")
                return stop_words
        except FileNotFoundError:
            logging.error(f"Stop words file not found: {file_path}")
            return set()

    def preprocess(self, text: str) -> Set[str]:
        text = self.extract_special_patterns(text)
        text = self.normalize_text(text)
        text = self.remove_special_characters(text)
        words = self.remove_stop_words(text)
        return {word for word in words if word.strip()}  # 过滤掉空字符串

    def extract_special_patterns(self, text: str) -> str:
        # 提取并删除文本中的特殊模式
        patterns = re.findall(r'([A-Za-z][^\w\s][A-Za-z])', text)
        for pattern in patterns:
            text = text.replace(pattern, '')
        return text

    def extract_ip_addresses(self, text: str) -> Set[str]:
        # 提取文本中的IP地址
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = set(re.findall(ip_pattern, text))
        if ips:
            logging.info(f"IP addresses extracted: {ips}")
        return ips

    def normalize_text(self, text: str) -> str:
        # 正规化文本，转换全角字符和空格
        normalized_text = ''.join(
            chr(ord(char) - 0xfee0 if 0xFF01 <= ord(char) <= 0xFF5E else 32 if ord(char) == 0x3000 else ord(char)) for
            char in text)
        logging.debug(f"Text normalized: {normalized_text}")
        return normalized_text

    def remove_special_characters(self, text: str) -> str:
        # 移除特殊字符和数字，合并空格
        text = re.sub(r'[^\w\s]', ' ', text)  # 替换所有非字母数字字符为空格
        text = re.sub(r'\d+', ' ', text)  # 删除所有数字
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        logging.debug(f"Special characters removed: {text}")
        return text

    def remove_stop_words(self, text: str) -> Set[str]:
        # 移除停用词
        words = jieba.cut(text)
        filtered_words = {word for word in words if word not in self.stop_words}
        logging.debug(f"Stop words removed. Words remaining: {filtered_words}")
        return filtered_words


@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    if not text:
        logging.error("No text provided in the request.")
        return jsonify({'error': 'No text provided'}), 400

    processor = InputProcessor(use_tfidf=True, ngram_range=(1, 2), keyword_threshold=0.2)
    processed_text = processor.preprocess(text)
    processed_text.add('%ip')  # 特殊键值添加
    response_data = {'processed_text': list(processed_text)}
    logging.info("Text processed successfully.")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
