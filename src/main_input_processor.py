# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/7/2 14:08
# @Function: 与Flask服务交互的外部脚本，用于处理文本预处理


import requests
import json

def process_text(input_text, url='http://127.0.0.1:5000/process'):
    """发送POST请求到Flask应用以进行文本预处理"""
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'text': input_text})
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': '连接服务器失败或处理请求失败', 'status_code': response.status_code}

if __name__ == '__main__':
    # 示例文本，检测前请替换
    text_to_check = "这是一个示例文本，请替换为实际需要处理的文本。"
    result = process_text(text_to_check)
    print("服务器响应：", result)
