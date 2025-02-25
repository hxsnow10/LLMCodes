#!/usr/bin/env python
# -*- encoding=utf8
# @author      : xiahong (xiahahaha01@gmail.com)
# @file        : predict
# @created     : Monday Feb 24, 2025 10:46:54 CST

"""


"""
import argparse
from transformers import pipeline

# 创建情感分析pipeline
classifier = pipeline('sentiment-analysis')

# 进行预测
result = classifier('We are very happy to introduce pipeline to the transformers repository.')
print(result)  # 输出预测结果

# 创建文本生成pipeline
generator = pipeline('text-generation')

# 生成文本
result = generator('Once upon a time', max_length=30)
print(result)


from transformers import AutoTokenizer, AutoModel

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 处理输入文本
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs)

def main():
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default=None, type=str)
    parser.add_argument("--foo", action = "store_true")
    args = parser.parse_args()

    main()

