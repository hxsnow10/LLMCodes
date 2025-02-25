#!/usr/bin/env python
# -*- encoding=utf8
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : predict_api
# @created     : Monday Feb 24, 2025 10:53:02 CST

"""
# 基本使用（输出到文件）
python openai_predict.py --prompt "什么是人工智能?" -o "./outputs/response.txt"

# 使用JSON格式输出
python openai_predict.py --prompt "解释深度学习" -o "./outputs/response.json" --format json

# 从文件读取提示并保存响应
python openai_predict.py -i input.txt -o "./outputs/response.txt"

"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pathlib import Path

class ChatCompletion:
    """OpenAI API调用封装类"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Optional[str]:
        """生成对话响应"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                collected_chunks = []
                # 处理流式响应
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        print(content, end='', flush=True)
                        collected_chunks.append(content)
                print()  # 添加最后的换行
                return ''.join(collected_chunks)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"API调用错误: {str(e)}", file=sys.stderr)
            return None

def save_response(content: str, output_path: str, format: str = 'txt') -> bool:
    """
    保存响应内容到文件
    
    Args:
        content: 响应内容
        output_path: 输出文件路径
        format: 输出格式 (txt 或 json)
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'content': content
            }
            output_path.write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2), 
                encoding='utf-8'
            )
        else:
            output_path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"保存文件失败: {str(e)}", file=sys.stderr)
        return False

def read_prompt_from_file(file_path: str) -> Optional[str]:
    """从文件读取提示文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取文件失败: {str(e)}", file=sys.stderr)
        return None

def main(
    input_path: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    output_path: Optional[str] = None,
    output_format: str = 'txt',
    stream: bool = False
) -> int:
    # 获取API密钥
    api_key = os.getenv("OPENAI_API_KEY") or "sk-31943b62c3b5411a86fdaa641aa2c624"
    if not api_key:
        print("错误: 未设置OPENAI_API_KEY环境变量", file=sys.stderr)
        return 1

    # 创建chat completion实例
    chat = ChatCompletion(api_key, base_url)

    # 确定输入提示
    if input_path:
        prompt = read_prompt_from_file(input_path)
        if not prompt:
            return 1
    elif not prompt:
        prompt = "请解释什么是机器学习?"

    # 生成响应
    response = chat.generate(
        prompt=prompt,
        model=model,
        temperature=temperature,
        stream=stream
    )

    if response:
        if not stream:
            print(f"AI响应:\n{response}")
        
        # 如果指定了输出路径，保存到文件
        if output_path:
            if save_response(response, output_path, output_format):
                print(f"响应已保存到: {output_path}")
            else:
                return 1
        return 0
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI API对话生成工具")
    parser.add_argument("-i", "--input_path", type=str, help="输入提示文件路径")
    parser.add_argument("--base_url", default = "https://api.deepseek.com", type=str, help="API基础URL")
    parser.add_argument("--prompt", type=str, help="直接输入提示文本")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="选择模型")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数(0-1)")
    parser.add_argument("-o", "--output_path", type=str, help="输出文件路径")
    parser.add_argument("-output_format", type=str, choices=['txt', 'json'], 
                       default='txt', help="输出格式(txt或json)")
    parser.add_argument("--stream", action="store_true", help="启用流式输出")
    
    args = parser.parse_args()
    sys.exit(main(**vars(args)))
