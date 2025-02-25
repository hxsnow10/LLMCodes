#!/usr/bin/env python
# -*- encoding=utf8
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : simple_api
# @created     : Monday Feb 24, 2025 11:25:54 CST

"""


"""
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os
import sys

# 科学上网反而连不上deepseek

exit_code = os.system('unset http_proxy && unset https_proxy')
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
proxy_api_key = os.get("PROXY_API_KEY")
lient = OpenAI(api_key = deepseek_api_key, base_url="https://api.deepseek.com")
# client = OpenAI(api_key=proxy_api_key, base_url="https://api.fast-tunnel.one/v1")
response = client.chat.completions.create(
    model="deepseek-chat",
    # model = "gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

try:
    # 创建流式响应
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "hello"},
        ],
        stream=True  # 启用流式输出
    )
    
    # 实时打印响应内容
    collected_chunks = []
    collected_messages = []
    
    # 处理流式响应
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            collected_chunks.append(chunk_message)
            # 实时打印，不换行
            print(chunk_message, end="\n", flush=True)
            collected_messages.append(chunk_message)
    
    # 打印换行
    print("\n")
    # 返回完整的响应文本
    full_reply = ''.join(collected_messages)
    print(full_reply)
    
except Exception as e:
    print(f"发生错误: {str(e)}", file=sys.stderr)

