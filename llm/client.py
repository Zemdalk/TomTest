"""
LLM 客户端

封装 LLM 调用逻辑，支持多种模型和配置。
支持批量并行调用、自动重试、结构化输出和 Mock 响应。

合并自:
- BenchRAG/benchrag/llm/api_llm.py
- Crux/src/crux/utils/llm_client.py
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import openai
from pydantic import BaseModel
from tqdm import tqdm


# 内部状态标记，用于 batch_call_json 重试逻辑
_SUCCESS = "S"
_FAILURE = "F"

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
class Generation:
    """Single LLM generation result."""

    text: str
    reasoning: str = ""


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
    统一的 LLM 客户端

    支持:
    - OpenAI 兼容 API
    - Mock 响应（用于测试）
    - JSON 格式输出
    - 结构化输出（Pydantic 模型）
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
        presence_penalty: float = 2.0,
        enable_thinking: bool = True,
        n_return: int = 1,
        max_workers: int = 32,
        mock_llm: bool = False,
    ):
        """Initialize the LLM client.

        Args:
            model_name: Model identifier
            api_key: API authentication key
            api_url: API base URL
            temperature: Sampling temperature (default: 0.6)
            max_tokens: Maximum completion tokens (default: 32768)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 20)
            presence_penalty: Presence penalty (default: 2.0)
            enable_thinking: Enable reasoning extraction (default: True)
            n_return: Number of completions per request (default: 1)
            max_workers: Max threads for batch operations (default: 32)
            mock_llm: Enable mock mode for testing (default: False)
        """
        self.model = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.mock_llm = mock_llm

        # 延迟初始化客户端
        self._client = None

        # Generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking
        self.n_return = n_return

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
            presence_penalty=config.get("presence_penalty", 2.0),
            enable_thinking=config.get("enable_thinking", True),
            n_return=config.get("n_return", 1),
            max_workers=config.get("max_workers", 32),
            mock_llm=config.get("mock_llm", False),
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
    # Mock Response
    # -----------------------------------------------------------------------

    def _mock_response(self, prompt: str) -> Dict[str, Any]:
        """Mock 响应 - 用于测试"""
        # Gap analysis
        if "Strategy Planner" in prompt or "信息覆盖" in prompt:
            return {"status": "sufficient", "missing_info": None}

        # Adjudication
        if "Critical Judge" in prompt or "研判" in prompt:
            return {
                "relevance": "Perfectly Relevant",
                "evidence": ["这是一篇关于信息检索的重要论文。", "提出了新颖的检索增强方法。"],
                "reason": "内容与查询直接相关",
            }

        # Intent parsing
        if "Intent Parsing" in prompt or "意图解析" in prompt:
            return {
                "user_goal": "FACTUAL",
                "constraints": {
                    "structured_metadata": [
                        {"field": "category", "operator": "in", "value": ["cs.IR", "cs.CL", "cs.AI"]}
                    ]
                },
                "keywords_bm25": ["信息检索", "RAG", "检索增强"],
                "queries_vector": ["检索增强生成技术研究", "信息检索智能体"],
                "rubric": "文档必须与信息检索或 RAG 技术相关",
                "missing_info_gap": None,
            }

        # Default
        return {
            "user_goal": "FACTUAL",
            "constraints": {"structured_metadata": []},
            "keywords_bm25": [],
            "queries_vector": [],
            "rubric": "相关内容",
            "missing_info_gap": None,
        }

    # -----------------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------------

    def _raw_generate(
        self,
        prompt: str,
        instruction: str,
        n: int,
        schema: Optional[dict] = None,
    ) -> Tuple[List[Generation], LLMUsage]:
        """Make a single API call to the LLM.

        Args:
            prompt: User prompt content
            instruction: System instruction
            n: Number of completions to generate
            schema: Optional JSON schema for structured output

        Returns:
            Tuple of (list of generations, usage statistics)
        """
        start = time.time()

        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=n,
            presence_penalty=self.presence_penalty,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            extra_body={"top_k": self.top_k},
        )

        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                },
            }

        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - start

        usage = LLMUsage(latency=latency)
        if getattr(resp, "usage", None):
            usage.prompt_tokens = resp.usage.prompt_tokens
            usage.completion_tokens = resp.usage.completion_tokens
            usage.total_tokens = resp.usage.total_tokens

        outputs = []
        for choice in resp.choices:
            msg = choice.message
            text = (msg.content or "").strip()

            reasoning = ""
            if self.enable_thinking:
                reasoning = (
                    getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                )

            outputs.append(Generation(text=text, reasoning=reasoning))

        return outputs, usage

    # -----------------------------------------------------------------------
    # Retry Wrapper
    # -----------------------------------------------------------------------

    def _generate_retry(
        self,
        prompt: str,
        instruction: str,
        n: int,
        schema: Optional[dict] = None,
        schema_model: Optional[Type[BaseModel]] = None,
        max_retry: int = 5,
    ) -> Tuple[List[Generation], LLMUsage]:
        """Generate with retry logic for transient failures.

        Args:
            prompt: User prompt content
            instruction: System instruction
            n: Number of completions to generate
            schema: Optional JSON schema for structured output
            schema_model: Optional Pydantic model for validation
            max_retry: Maximum retry attempts

        Returns:
            Tuple of (list of generations, usage statistics)
        """
        last_usage = LLMUsage()

        for _ in range(max_retry):
            try:
                outputs, usage = self._raw_generate(prompt, instruction, n, schema)
                last_usage = usage

                # Check if any output has non-empty text
                if not any(o.text for o in outputs):
                    continue

                # If no schema, return immediately
                if schema is None:
                    return outputs, usage

                # Validate JSON and schema if provided
                for output in outputs:
                    try:
                        data = json.loads(output.text)
                        if schema_model is not None:
                            schema_model.model_validate(data)
                        return outputs, usage
                    except (json.JSONDecodeError, Exception):
                        continue

            except Exception as e:
                import traceback
                logging.warning(f"[LLM] generate failed (will retry): {e}\n{traceback.format_exc()}")

        logging.error(f"[LLM] all {max_retry} retries exhausted, returning empty generation.")
        return [Generation("")], last_usage

    # -----------------------------------------------------------------------
    # Public Text API
    # -----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        instruction: str = "You are a helpful assistant.",
        n: Optional[int] = None,
    ) -> Tuple[List[Generation], LLMUsage]:
        """Generate text from a single prompt.

        Args:
            prompt: User prompt content
            instruction: System instruction
            n: Number of completions (default: self.n_return)

        Returns:
            Tuple of (list of generations, usage statistics)
        """
        n = n or self.n_return
        outputs, usage = self._generate_retry(prompt, instruction, n)

        success = any(o.text for o in outputs)
        self._track_usage(usage, success)

        return outputs, usage

    def batch_generate(
        self,
        prompts: List[str],
        instructions: Optional[List[str]] = None,
        n: Optional[int] = None,
    ) -> List[Tuple[List[Generation], LLMUsage]]:
        """Generate text from multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            instructions: List of system instructions (default: all use default instruction)
            n: Number of completions per prompt (default: self.n_return)

        Returns:
            List of (generations, usage) tuples
        """
        n = n or self.n_return
        if instructions is None:
            instructions = ["You are a helpful assistant."] * len(prompts)

        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self.generate, p, i, n) for p, i in zip(prompts, instructions)
            ]

            results = []
            for future in tqdm(futures, total=len(futures), desc="Generating"):
                results.append(future.result())

            return results

    # -----------------------------------------------------------------------
    # JSON Output API
    # -----------------------------------------------------------------------

    def call_json(
        self,
        prompt: str,
        instruction: str = "You are a helpful assistant.",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        调用 LLM 并返回 JSON 结果

        Args:
            prompt: 提示词
            instruction: 系统指令
            model: 可选的模型名称

        Returns:
            解析后的 JSON 对象
        """
        if self.mock_llm:
            return self._mock_response(prompt)

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logging.info(f"[LLM] 调用失败: {e}")
            if self.mock_llm:
                return self._mock_response(prompt)
            raise

    def _call_json_with_status(
        self,
        prompt: str,
        instruction: str,
        model: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        内部方法：调用 LLM 并返回 (结果, 状态) 元组

        用于 batch_call_json 的重试逻辑
        """
        if self.mock_llm:
            return (self._mock_response(prompt), _SUCCESS)

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return (json.loads(content), _SUCCESS)
        except Exception as e:
            logging.info(f"[LLM] 调用失败: {e}")
            return (self._mock_response(prompt), _FAILURE)

    def batch_call_json(
        self,
        prompts: List[str],
        instructions: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_retry: int = 5,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量调用 LLM 并返回 JSON 结果

        支持并行调用和自动重试失败的请求。

        Args:
            prompts: 提示词列表
            instructions: 系统指令列表 (默认: 所有使用默认指令)
            model: 可选的模型名称
            max_retry: 最大重试次数
            max_workers: 并发线程数 (默认: self.max_workers)

        Returns:
            结果列表，顺序与输入 prompts 一致。
            每个元素为解析后的 JSON 对象。
        """
        if instructions is None:
            instructions = ["You are a helpful assistant."] * len(prompts)

        max_workers = max_workers or self.max_workers

        if self.mock_llm:
            return [self._mock_response(prompt) for prompt in prompts]

        # 内部使用带状态的结果进行重试跟踪
        results_with_status: List[Optional[Tuple[Dict[str, Any], str]]] = [None] * len(prompts)
        idxs_to_retry = list(range(len(prompts)))
        current_prompts = prompts
        current_instructions = instructions
        retry_count = 0

        while idxs_to_retry and retry_count < max_retry:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._call_json_with_status, current_prompts[i], current_instructions[i], model
                    ): i
                    for i in range(len(current_prompts))
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing (retry {retry_count})",
                ):
                    sub_idx = futures[future]
                    idx = idxs_to_retry[sub_idx]
                    results_with_status[idx] = future.result()

            # 收集失败的请求索引，准备重试
            idxs_to_retry = [
                i for i, x in enumerate(results_with_status) if x and x[1] == _FAILURE
            ]
            current_prompts = [prompts[i] for i in idxs_to_retry]
            current_instructions = [instructions[i] for i in idxs_to_retry]
            retry_count += 1

        # 提取结果（去除状态标记）
        return [r[0] if r else {} for r in results_with_status]

    # -----------------------------------------------------------------------
    # Structured Output API
    # -----------------------------------------------------------------------

    def generate_structured(
        self,
        prompt: str,
        instruction: str,
        schema_model: Type[BaseModel],
        n: Optional[int] = None,
    ) -> Tuple[List[BaseModel], LLMUsage]:
        """Generate structured output matching a Pydantic model.

        Args:
            prompt: User prompt content
            instruction: System instruction
            schema_model: Pydantic model for validation
            n: Number of completions (default: self.n_return)

        Returns:
            Tuple of (list of validated models, usage statistics)
        """
        n = n or self.n_return
        schema = schema_model.model_json_schema()

        outputs, usage = self._generate_retry(
            prompt,
            instruction,
            n,
            schema=schema,
            schema_model=schema_model,
        )

        results = []
        success = False
        for output in outputs:
            try:
                data = json.loads(output.text)
                validated = schema_model.model_validate(data)
                results.append(validated)
                success = True  # 只要有一个成功，记作成功
            except (json.JSONDecodeError, Exception):
                pass

        # 更新全局 usage
        self._track_usage(usage, success)

        return results, usage

    def batch_generate_structured(
        self,
        prompts: List[str],
        instructions: List[str],
        schema_model: Type[BaseModel],
        n: Optional[int] = None,
    ) -> List[List[BaseModel]]:
        """Generate structured outputs from multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            instructions: List of system instructions
            schema_model: Pydantic model for validation
            n: Number of completions per prompt (default: self.n_return)

        Returns:
            List of lists of validated models
        """
        n = n or self.n_return

        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_structured, p, i, schema_model, n)
                for p, i in zip(prompts, instructions)
            ]

            results = []
            for future in tqdm(futures, total=len(futures), desc="Generating structured"):
                validated, _ = future.result()  # usage 已在 generate_structured 内更新
                results.append(validated)

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}', mock={self.mock_llm})"
