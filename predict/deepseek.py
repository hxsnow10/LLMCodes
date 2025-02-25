#!/usr/bin/env python
# -*- encoding=utf8
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : deepseek
# @created     : Monday Feb 24, 2025 14:33:48 CST

"""DeepSeek模型预测器
"""


import os
import sys
import argparse
from typing import Optional, List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    TextIteratorStreamer
)
import torch
from threading import Thread

class DeepSeekPredictor:
    """DeepSeek模型预测器"""
    
    def __init__(
        self, 
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
        cache_dir: str = "./models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        stream: bool = True
    ) -> str:
        """生成回答"""
        # 设置生成配置
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if stream:
            # 创建流式输出
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(
                **inputs,
                generation_config=generation_config,
                streamer=streamer,
            )
            
            # 在新线程中生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 实时输出生成内容
            generated_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                generated_text += new_text
            print()  # 最后打印换行
            
            return generated_text
        else:
            # 非流式生成
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

import os
import sys
import argparse
from typing import Optional
from PIL import Image
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    GenerationConfig
)

class DeepSeekVisionPredictor:
    """DeepSeek视觉模型预测器"""
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl-7b-base",
        cache_dir: str = "./models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载处理器和模型
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def describe_image(
        self,
        image_path: str,
        prompt: str = "请描述这张图片",
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """生成图片描述"""
        try:
            # 加载和预处理图像
            image = Image.open(image_path)
            
            # 准备输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # 设置生成配置
            generation_config = GenerationConfig(
                temperature=temperature,
                max_new_tokens=max_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # 生成描述
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            
            # 解码输出
            description = self.processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            return description
            
        except Exception as e:
            print(f"处理图像时出错: {str(e)}", file=sys.stderr)
            return None

def main():
    parser = argparse.ArgumentParser(description="DeepSeek视觉模型图像描述工具")
    parser.add_argument("-i", "--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--prompt", type=str, default="请描述这张图片", help="提示文本")
    parser.add_argument("--cache-dir", type=str, default="./models", help="模型缓存目录")
    parser.add_argument("--max-length", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    args = parser.parse_args()

    try:
        # 初始化预测器
        predictor = DeepSeekVisionPredictor(cache_dir=args.cache_dir)
        
        # 生成图片描述
        description = predictor.describe_image(
            image_path=args.image,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        if description:
            print(f"\n图片描述:\n{description}")
            return 0
        return 1
        
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DeepSeek模型预测工具")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示")
    parser.add_argument("--cache-dir", type=str, default="./models", help="模型缓存目录")
    parser.add_argument("--max-length", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    args = parser.parse_args()

    try:
        # 初始化预测器
        predictor = DeepSeekPredictor(cache_dir=args.cache_dir)
        
        # 生成回答
        response = predictor.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            stream=not args.no_stream
        )
        
        return 0
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

