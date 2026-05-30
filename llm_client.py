import os
import json
import time
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

class LLMClient:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages, stream=False):
        """基本调用与流式输出，包含错误处理与重试"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=stream,
                    temperature=0.7
                )
                if stream:
                    return self._stream_generator(response)
                return response.choices[0].message.content
            except (RateLimitError, APITimeoutError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"API 请求失败 (重试{max_retries}次后): {str(e)}")
            except APIError as e:
                raise Exception(f"API 鉴权或网络错误: {str(e)}")

    def _stream_generator(self, response):
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# 模型配置映射
MODEL_CONFIGS = {
    "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "Qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-turbo"},
    "Kimi": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "Local-Ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2:7b"}
}
