"""
LLM 客户端

简化的 LLM 调用封装，只支持两个核心功能：
1. 文本生成 - 使用 chat.completions.create
2. 结构化输出 - 使用 chat.completions.parse
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import openai
from pydantic import BaseModel
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    """Token usage and latency for a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    简化的 LLM 客户端

    支持:
    - 文本生成: generate(), batch_generate()
    - 结构化输出: generate_structure(), batch_generate_structure()
    - 批量并行调用与自动重试
    - 线程安全的使用统计
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_url: str,
        temperature: float = 0.6,
        max_tokens: int = 32768,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 2,
        enable_thinking: bool = True,
        max_workers: int = 32,
    ):
        """Initialize LLM client.

        Args:
            model_name: Model identifier
            api_key: API authentication key
            api_url: API base URL
            temperature: Sampling temperature (default: 0.6)
            max_tokens: Maximum completion tokens (default: 32768)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 20)
            presence_penalty: Presence penalty (default: 2)
            enable_thinking: Enable thinking mode (default: True)
            max_workers: Max threads for batch operations (default: 32)
        """
        self.model = model_name
        self.api_key = api_key
        self.api_url = api_url

        # 延迟初始化客户端
        self._client = None

        # Generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking

        # Batch processing
        self.max_workers = max_workers

        # Usage tracking (thread-safe)
        self.usage: Dict[str, Any] = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
        }
        self._lock = threading.Lock()

        # parse API 支持检测（首次尝试失败即标记）
        self._parse_supported = None  # None: 未尝试, True: 支持, False: 不支持
        self._parse_lock = threading.Lock()

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.api_url)
        return self._client

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """从配置字典创建客户端实例"""
        return cls(
            model_name=config["model_name"],
            api_key=config["api_key"],
            api_url=config["api_url"],
            temperature=config.get("temperature", 0.6),
            max_tokens=config.get("max_tokens", 32768),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 20),
            presence_penalty=config.get("presence_penalty", 2),
            enable_thinking=config.get("enable_thinking", True),
            max_workers=config.get("max_workers", 32),
        )

    # -----------------------------------------------------------------------
    # Usage Tracking (Thread-Safe)
    # -----------------------------------------------------------------------

    def _track_usage(self, usage: LLMUsage, success: bool = True) -> None:
        """Update global usage statistics in a thread-safe manner."""
        with self._lock:
            self.usage["total_prompt_tokens"] += usage.prompt_tokens
            self.usage["total_completion_tokens"] += usage.completion_tokens
            self.usage["total_tokens"] += usage.total_tokens
            self.usage["total_latency"] += usage.latency
            self.usage["total_calls"] += 1
            if success:
                self.usage["successful_calls"] += 1
            else:
                self.usage["failed_calls"] += 1

    def get_usage(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        with self._lock:
            return dict(self.usage)

    def reset_usage(self) -> None:
        """重置使用统计"""
        with self._lock:
            self.usage = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
            }

    # -----------------------------------------------------------------------
    # Core API: Text Generation
    # -----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_retry: int = 5,
    ) -> str:
        """Generate text from a single prompt with retry.

        Args:
            prompt: User prompt content
            max_retry: Maximum retry attempts

        Returns:
            Generated text, empty string if all retries fail
        """
        # 构建 extra_body
        extra_body: Dict[str, Any] = {"top_k": self.top_k}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(max_retry):
            try:
                start = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    extra_body=extra_body,
                )

                latency = time.time() - start
                usage = LLMUsage(latency=latency)
                if hasattr(response, "usage") and response.usage:
                    usage.prompt_tokens = response.usage.prompt_tokens
                    usage.completion_tokens = response.usage.completion_tokens
                    usage.total_tokens = response.usage.total_tokens

                text = response.choices[0].message.content or ""
                if text:
                    self._track_usage(usage, success=True)
                    return text

            except Exception as e:
                import traceback
                logging.warning(f"[LLM] generate attempt {attempt + 1}")

        logging.error(f"[LLM] generate all {max_retry} attempts exhausted, returning empty.")
        self._track_usage(LLMUsage(), success=False)
        return ""

    def batch_generate(
        self,
        prompts: List[str],
    ) -> List[str]:
        """Generate text from multiple prompts in parallel.

        Args:
            prompts: List of user prompts

        Returns:
            List of generated texts
        """
        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self.generate, p)
                for p in prompts
            ]

            results = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc="Generating",
                miniters=100,
            ):
                results.append(future.result())

            return results

    # -----------------------------------------------------------------------
    # Core API: Structured Output (parse)
    # -----------------------------------------------------------------------

    def generate_structure(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
    ) -> BaseModel:
        """调用 LLM，返回 Pydantic 对象（自动适配不同模型）。

        两阶段降级策略：
        1. 首选：chat.completions.parse() - 直接返回 Pydantic 对象，最佳体验
        2. 降级：chat.completions.create() + json_object response_format + prompt 引导 + 解析验证

        Args:
            prompt: 提示词
            response_object: 必须提供的 Pydantic 模型类
            max_retry: 最大重试次数

        Returns:
            response_object 的实例，失败时返回空实例
        """
        # 首次检测：尝试使用 parse API
        if self._parse_supported is None:
            with self._parse_lock:
                if self._parse_supported is None:
                    # 尝试一次，成功则标记支持，失败则不支持
                    try:
                        result = self._generate_with_parse(prompt, response_object, max_retry=1)
                        self._parse_supported = True
                        return result
                    except Exception:
                        self._parse_supported = False
                        logging.warning(f"[LLM] Model {self.model} parse API failed, switching to JSON object mode")
                        # 继续使用降级模式

        # 根据检测结果选择模式
        if self._parse_supported:
            return self._generate_with_parse(prompt, response_object, max_retry)
        else:
            return self._generate_with_json_object(prompt, response_object, max_retry)

    def _generate_with_parse(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
    ) -> BaseModel:
        """使用 parse API 的原生结构化输出。"""
        extra_body: Dict[str, Any] = {"top_k": self.top_k}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(max_retry):
            try:
                start = time.time()
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    response_format=response_object,
                    extra_body=extra_body,
                )
                latency = time.time() - start
                usage = LLMUsage(latency=latency)
                if hasattr(response, "usage") and response.usage:
                    usage.prompt_tokens = response.usage.prompt_tokens
                    usage.completion_tokens = response.usage.completion_tokens
                    usage.total_tokens = response.usage.total_tokens

                result = response.choices[0].message.parsed
                self._track_usage(usage, success=True)
                return result

            except Exception as e:
                import traceback
                logging.warning(f"[LLM] parse mode attempt {attempt + 1}")

        logging.error(f"[LLM] parse mode all {max_retry} attempts exhausted")
        self._track_usage(LLMUsage(), success=False)
        return response_object.model_construct()

    def _generate_with_json_object(
        self,
        prompt: str,
        response_object: Type[BaseModel],
        max_retry: int = 5,
    ) -> BaseModel:
        """降级模式：使用 json_object response_format + prompt 引导 + 解析验证。"""
        import json

        # 构建 schema 描述
        schema_desc = self._format_schema_for_prompt(response_object)
        # 增强提示词
        enhanced_prompt = f"""{prompt}

---
Response Format:
{schema_desc}

IMPORTANT: Respond with a valid JSON object that matches the schema above.
Output ONLY the JSON object, no additional text or markdown formatting."""

        extra_body: Dict[str, Any] = {"top_k": self.top_k}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        for attempt in range(max_retry):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    #response_format={"type": "json_object"},  # 使用 json_object
                    extra_body=extra_body,
                )

                latency = time.time() - start
                usage = LLMUsage(latency=latency)
                if hasattr(response, "usage") and response.usage:
                    usage.prompt_tokens = response.usage.prompt_tokens
                    usage.completion_tokens = response.usage.completion_tokens
                    usage.total_tokens = response.usage.total_tokens

                content = response.choices[0].message.content or ""
                # 提取 JSON
                json_data = self._extract_json(content)
                if json_data is None:
                    raise ValueError(f"Failed to extract valid JSON: {content[:200]}")

                # 用 Pydantic 验证（不符合就重试）
                result = response_object.model_validate(json_data)
                self._track_usage(usage, success=True)
                return result

            except Exception as e:
                import traceback
                logging.warning(f"[LLM] json_object mode attempt {attempt + 1}")

        logging.error(f"[LLM] json_object mode all {max_retry} attempts exhausted")
        self._track_usage(LLMUsage(), success=False)
        return response_object.model_construct()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取 JSON。"""
        import json
        import re

        text = text.strip()

        # 尝试 1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试 2: 提取 ```json ... ``` 代码块
        matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # 尝试 3: 提取 {} 大括号内容
        matches = re.findall(r'\{[\s\S]*?\}', text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def _format_schema_for_prompt(self, response_object: Type[BaseModel]) -> str:
        """将 Schema 格式化为描述。"""
        schema_dict = response_object.model_json_schema()
        properties = schema_dict.get("properties", {})
        required = schema_dict.get("required", [])

        lines = ["Schema:"]
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "any")
            is_required = field_name in required
            enum = field_schema.get("enum")

            if enum:
                type_desc = f"one of: {', '.join(repr(v) for v in enum)}"
            else:
                type_desc = field_type

            desc = field_schema.get("description", "")
            req_str = " (required)" if is_required else " (optional)"

            lines.append(f"  {field_name}: {type_desc}{req_str} - {desc}")

        return "\n".join(lines)

    def batch_generate_structure(
        self,
        prompts: List[str],
        response_object: Type[BaseModel],
    ) -> List[BaseModel]:
        """
        批量调用 LLM，返回 Pydantic 对象列表。

        支持并行调用。

        Args:
            prompts: 提示词列表
            response_object: 必须提供的 Pydantic 模型类

        Returns:
            response_object 实例的列表
        """
        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_structure, p, response_object)
                for p in prompts
            ]

            results = []
            for future in tqdm(
                futures,
                total=len(futures),
                desc="Generating structure",
                miniters=100,
            ):
                results.append(future.result())

            return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
