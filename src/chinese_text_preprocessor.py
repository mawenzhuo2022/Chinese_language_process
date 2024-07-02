# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/05/21 10:17
# @Function: 实现一个中文文本预处理类，支持分词、去除停用词、标准化、去除数字、向量化和关键词提取。

import jieba
import re
import logging
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Set, Tuple

# 配置日志，方便调试和查看程序运行状态
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class chinese_text_preprocessor:
    """中文文本预处理类"""

    def __init__(self, stop_words_file: str = '../dat/stop_words/stop_words_full.txt', use_tfidf: bool = True,
                 ngram_range: Tuple[int, int] = (1, 1), keyword_threshold: float = 0.1):
        """初始化函数
        :param stop_words_file: 停用词文件路径
        :param use_tfidf: 是否使用TF-IDF向量化，默认为True
        :param ngram_range: n-gram范围，默认为(1, 1)仅单个词
        :param keyword_threshold: 提取关键词的TF-IDF阈值
        """
        self.stop_words = self.load_stop_words(stop_words_file)
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=self.stop_words,
                                          ngram_range=ngram_range) if use_tfidf else CountVectorizer(
            tokenizer=jieba.cut, stop_words=self.stop_words, ngram_range=ngram_range)
        self.keyword_threshold = keyword_threshold
        logging.info("文本预处理器已初始化，使用TF-IDF: {}, ngram范围: {}, 关键词阈值: {}".format(use_tfidf, ngram_range,
                                                                                                 keyword_threshold))

    def load_stop_words(self, file_path: str) -> Set[str]:
        """加载停用词文件
        :param file_path: 停用词文件路径
        :return: 停用词集合
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                stop_words = {line.strip() for line in file}
                logging.info("加载停用词数量: {}".format(len(stop_words)))
                return stop_words
        except FileNotFoundError:
            logging.error("未找到停用词文件: {}".format(file_path))
            return set()

    def preprocess(self, text: str) -> Set[str]:
        """对文本进行预处理，包括标准化、去除特殊字符和停用词
        :param text: 原始文本
        :return: 预处理后的文本集合
        """
        special_patterns = self.extract_special_patterns(text)
        text = self.cleaned_text  # 使用提取特殊模式后清理的文本
        text = self.normalize_text(text)
        text = self.remove_special_characters(text)
        words = self.remove_stop_words(text)
        # 去除空字符串
        words = {word for word in words if word.strip()}
        # 加入特殊模式
        words.update(special_patterns)
        return words

    def extract_special_patterns(self, text: str) -> Set[str]:
        """提取特定模式如'I/O'并从文本中删除
        :param text: 原始文本
        :return: 特定模式集合
        """
        patterns = re.findall(r'([A-Za-z][^\w\s][A-Za-z])', text)
        # 从文本中删除这些模式
        for pattern in patterns:
            text = text.replace(pattern, '')
        self.cleaned_text = text  # 保留清理后的文本以供后续处理
        return set(patterns)

    def normalize_text(self, text: str) -> str:
        """文本标准化，包括全角到半角的转换
        :param text: 原始文本
        :return: 标准化后的文本
        """
        text = self.full_to_half(text)
        return text

    def remove_special_characters(self, text: str) -> str:
        """去除文本中的特殊字符和数字
        :param text: 标准化后的文本
        :return: 清理特殊字符和数字后的文本
        """
        text = re.sub(r'[^\w\s]', ' ', text)  # 替换所有非字母数字字符为空格
        text = re.sub(r'\d+', ' ', text)  # 删除所有数字
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        return text

    def remove_stop_words(self, text: str) -> Set[str]:
        """去除停用词
        :param text: 清理特殊字符后的文本
        :return: 去除停用词后的文本集合
        """
        words = jieba.cut(text)
        filtered_words = {word for word in words if word not in self.stop_words}
        return filtered_words

    @staticmethod
    def full_to_half(s: str) -> str:
        """全角字符转半角字符
        :param s: 原始字符串
        :return: 转换后的字符串
        """
        return ''.join(
            chr(ord(char) - 0xfee0 if 0xFF01 <= ord(char) <= 0xFF5E else 32 if ord(char) == 0x3000 else ord(char)) for
            char in s)

    def extract_keywords(self, tfidf_matrix, feature_names) -> Set[str]:
        """提取关键词
        :param tfidf_matrix: TF-IDF向量化后的矩阵
        :param feature_names: 矩阵中的特征名（词汇）
        :return: 关键词集合
        """
        keywords_set = set()
        for row in tfidf_matrix:
            keywords_set.update(feature_names[col] for col in row.nonzero()[1] if row[0, col] > self.keyword_threshold)
        return keywords_set


# 使用示例
if __name__ == "__main__":
    input_file_path = '../dat/raw_data/testdata.csv'
    output_file_path = '../dat/words_set/words_set.csv'

    # 读取CSV文件中的问题
    questions = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if row:
                questions.append(row[1].strip())  # 提取每行的问题描述

    # 实例化预处理器
    preprocessor = chinese_text_preprocessor(use_tfidf=True, ngram_range=(1, 2), keyword_threshold=0.2)

    # 对每个问题进行预处理并存储结果
    with open(output_file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Processed Words Set'])  # 写入标题行
        for question in questions:
            words_set = preprocessor.preprocess(question)
            writer.writerow([question, ' '.join(words_set)])  # 将每个问题和其处理后的词汇集写入CSV
