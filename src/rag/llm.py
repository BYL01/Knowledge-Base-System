"""OpenAI 兼容的 Chat 客户端。DeepSeek / OpenAI / 任意兼容网关都走这一层。

DeepSeek 没有 embedding 接口，所以向量化仍用本地 hashing；
DeepSeek 用在生成和 LLM 判官两处。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}


class LLMError(RuntimeError):
    """调用失败。判官层会捕获它并降级，不让整批跑批挂掉。"""


@dataclass
class ChatResult:
    text: str
    model: str
    latency_ms: int


class ChatClient:
    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"未支持的 provider：{provider}")
        spec = PROVIDERS[provider]
        self.provider = provider
        self.model = model or spec["default_model"]
        self.base_url = (base_url or os.environ.get(f"{provider.upper()}_BASE_URL") or spec["base_url"]).rstrip("/")
        self.api_key = api_key or os.environ.get(spec["api_key_env"])
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None) -> ChatResult:
        if not self.api_key:
            raise LLMError(f"缺少 API key，请设置 {PROVIDERS[self.provider]['api_key_env']}")

        import requests  # 局部导入，不用这个后端时不产生依赖

        payload: dict[str, object] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                text = body["choices"][0]["message"]["content"].strip()
                latency_ms = int((time.perf_counter() - started) * 1000)
                return ChatResult(text=text, model=body.get("model", self.model), latency_ms=latency_ms)
            except Exception as exc:  # noqa: BLE001 - 统一收敛成 LLMError，交给上层降级
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        raise LLMError(f"{self.provider} 调用失败：{last_error}") from last_error
