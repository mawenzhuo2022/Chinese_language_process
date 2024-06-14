# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/5/22
# @Function: 主程序，连接文本预处理和相似问题查找功能

import sys
from input_processor import main as preprocess_main  # 假设这是文本预处理模块中的主函数
from vectorize import main as find_main  # 假设这是相似问题查找模块中的主函数

# main_find_similar.py

def main():

    # 文本预处理
    processed_input_set = preprocess_main()

    # 将处理后的集合转换为单一字符串
    processed_input = ' '.join(processed_input_set)

    # 寻找相似问题
    find_main(processed_input)

if __name__ == "__main__":
    main()
