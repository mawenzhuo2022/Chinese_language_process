# Chinese Text Preprocessor README

## 概述 Overview

这个项目提供了一个用于预处理中文文本数据的Python类 `chinese_text_preprocessor`。它支持多种操作，包括分词、去除停用词、文本标准化、特殊字符和数字去除、向量化以及关键词提取。该模块使用 `jieba` 进行分词，使用 `sklearn` 进行向量化。

This project provides a Python class `chinese_text_preprocessor` designed for preprocessing Chinese text data. It supports a variety of operations including tokenization, stop word removal, text normalization, special character and digit removal, vectorization, and keyword extraction. The module utilizes `jieba` for tokenization and `sklearn` for vectorization.

## 功能 Features

- **分词 Tokenization**: 将中文文本分割成单个词语。
- **去除停用词 Stop Word Removal**: 使用可定制的停用词列表过滤掉不必要的词语。
- **文本标准化 Text Normalization**: 将全角字符转换为半角。
- **去除特殊字符和数字 Removal of Special Characters and Digits**: 清理文本中的非字母数字字符和数字。
- **向量化 Vectorization**: 使用TF-IDF或计数向量化将文本转换为数值数据。
- **关键词提取 Keyword Extraction**: 基于TF-IDF值识别和提取关键词。

## 环境要求 Requirements

- Python 3.x
- `jieba`
- `sklearn`
- `re`
- `logging`

## 安装 Installation

在使用 `chinese_text_preprocessor` 前，请确保已安装Python和pip。可以使用pip安装所需的包：

Before using the `chinese_text_preprocessor`, make sure you have Python and pip installed. You can install the required packages using pip:

```bash
pip install jieba
pip install scikit-learn
```

## 使用说明 Usage

要使用 `chinese_text_preprocessor`，首先需要用适当的配置实例化它。

To use the `chinese_text_preprocessor`, you must first instantiate it with appropriate configurations.

运行 `main.py` 文件，程序将提示您输入文本，输入后按回车即可看到处理结果，结果将被打印在日志里。

Run the `main.py` file, the program will prompt you to enter text, press Enter after input, and the results will be printed in the log.

```python
# 运行主程序
# Run the main program
python main.py
```

## 配置 Configuration

- `stop_words_file`: 包含停用词的文件路径。
- `use_tfidf`: 布尔值，指示是否使用TF-IDF向量化；否则使用计数向量化。
- `ngram_range`: 表示要考虑的n-gram范围的元组。
- `keyword_threshold`: 基于TF-IDF确定关键词重要性的阈值。

## 示例应用 Example Application

该仓库包括一个示例应用，其中使用预处理器清洗并分析CSV文件中的问题。处理结果存储回另一个CSV中以供进一步分析。

The repository includes an example application where the preprocessor is used to clean and analyze questions from a CSV file. The processed results are stored back into another CSV for further analysis.

## 贡献 Contributing

欢迎对这个项目进行贡献。请fork仓库并提交带有您改进的pull request。

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## 许可 License

该项目采用MIT许可证授权 - 详情见LICENSE文件。

This project is licensed under the MIT License - see the LICENSE file for details.