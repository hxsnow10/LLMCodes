#!/usr/bin/env python
# -*- encoding=utf8
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : deepseek_api
# @created     : Monday Feb 24, 2025 19:53:28 CST

"""Deepseek API使用示例
"""

import argparse
import os,sys
from typing import Dict, List, Any
from openai import OpenAI

class DeepseekClient:
    def __init__(self, api_key: str) -> None:
        """初始化不同用途的客户端"""
        self.base_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.beta_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/beta")

def use_reasoning_model(client: OpenAI) -> None:
    """使用推理模型进行对话"""
    # Round 1
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )
    
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    print(f"Reasoning content: {reasoning_content}")
    print(f"Content: {content}")
    
    # Round 2
    messages.extend([
        {'role': 'assistant', 'content': content},
        {'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"}
    ])
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )
    print(f"Round 2 response: {response.choices[0].message.content}")

def chat_conversation(client: OpenAI) -> None:
    """进行多轮对话"""
    messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
    
    # Round 1
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    messages.append(response.choices[0].message)
    print(f"Messages Round 1: {messages}")
    
    # Round 2
    messages.append({"role": "user", "content": "What is the second?"})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    messages.append(response.choices[0].message)
    print(f"Messages Round 2: {messages}")

def prefix_code_completion(client: OpenAI) -> None:
    """对话前缀续写代码"""
    messages = [
        {"role": "user", "content": "Please write quick sort code"},
        {"role": "assistant", "content": "```python\n", "prefix": True}
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stop=["```"],
    )
    print(response.choices[0].message.content)

def fim_completion(client: OpenAI) -> None:
    """Fill-in-the-Middle (FIM) 补全"""
    response = client.completions.create(
        model="deepseek-chat",
        prompt="def fib(a):",
        suffix="    return fib(a-1) + fib(a-2)",
        max_tokens=128
    )
    print(response.choices[0].text)

def weather_function_calling(client: OpenAI) -> None:
    """天气查询的Function Calling示例"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user should supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]
    
    def send_messages(messages: List[Dict[str, Any]]) -> Any:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )
        return response.choices[0].message
    
    messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
    message = send_messages(messages)
    print(f"User>\t {messages[0]['content']}")
    
    tool = message.tool_calls[0]
    messages.append(message)
    messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
    message = send_messages(messages)
    print(f"Model>\t {message.content}")

def main(args: argparse.Namespace) -> int:
    """主函数"""
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable")
        
    # 创建DeepseekClient实例
    deepseek = DeepseekClient(api_key)
    
    # 使用基础客户端的功能
    use_reasoning_model(deepseek.base_client)
    chat_conversation(deepseek.base_client)
    weather_function_calling(deepseek.base_client)
    
    # 使用beta客户端的功能
    prefix_code_completion(deepseek.beta_client)
    fim_completion(deepseek.beta_client)
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    main(args)