"""LLM 接入层测试：用本地假服务端验证 HTTP 调用、判官缓存与熔断，不碰真实 API。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.rag.judge import LLMJudge
from src.rag.llm import ChatClient, LLMError


class _FakeHandler(BaseHTTPRequestHandler):
    """按 prompt 内容返回固定判官结论的假 DeepSeek 服务端。"""

    calls = 0

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 接口
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        prompt = payload["messages"][0]["content"]
        type(self).calls += 1

        if "花呗" in prompt:
            content = '```json\n{"answerable": false, "reason": "资料中未涉及花呗支付"}\n```'
        else:
            content = '{"answerable": true, "reason": "资料中有明确依据"}'

        body = json.dumps({"model": "deepseek-chat", "choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静音
        return


@pytest.fixture(scope="module")
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture
def client(fake_server):
    return ChatClient(provider="deepseek", base_url=fake_server, api_key="test-key", timeout=5)


def test_chat_client_parses_response(client):
    result = client.chat("资料里没有提到花呗")
    assert result.model == "deepseek-chat"
    assert "answerable" in result.text


def test_judge_extracts_json_from_code_block(client, tmp_path):
    judge = LLMJudge(client, cache_dir=tmp_path / "cache")
    result = judge.judge_answerability("可以用花呗分期支付吗", ["支持微信支付、支付宝、银联在线。"])
    assert result.answerable is False
    assert result.source == "llm"
    assert "花呗" in result.reason


def test_judge_cache_avoids_second_call(client, tmp_path):
    _FakeHandler.calls = 0
    judge = LLMJudge(client, cache_dir=tmp_path / "cache")
    query = "生鲜商品支持七天无理由退货吗"
    contexts = ["生鲜商品不支持 7 天无理由退货，但品质问题可全额赔付。"]

    first = judge.judge_answerability(query, contexts)
    second = judge.judge_answerability(query, contexts)

    assert first.answerable is True
    assert second.answerable is True
    assert second.source == "cache"
    assert _FakeHandler.calls == 1, "命中缓存后不应再发请求"


def test_judge_circuit_breaker_after_failures():
    """判官连不上时要熔断，不能让整批跑批卡死。"""
    dead = ChatClient(provider="deepseek", base_url="http://127.0.0.1:1/v1", api_key="k", timeout=1, max_retries=0)
    judge = LLMJudge(dead, on_error="fallback", failure_circuit=3)

    for _ in range(3):
        result = judge.judge_answerability("随便问问", ["一些资料"])
        assert result.source == "fallback"
        assert result.answerable is True  # fallback = 按可答处理，等价于基线行为

    assert judge._tripped is True
    final = judge.judge_answerability("再问一次", ["一些资料"])
    assert final.error and "熔断" in final.error


def test_judge_cache_only_mode_without_cache(tmp_path):
    """无外网机器用 cache_only：没有缓存就按兜底策略走，不发请求。"""
    dead = ChatClient(provider="deepseek", base_url="http://127.0.0.1:1/v1", api_key="k", timeout=1, max_retries=0)
    judge = LLMJudge(dead, cache_dir=tmp_path / "cache", cache_only=True, on_error="refuse")
    result = judge.judge_answerability("没有缓存的问题", ["一些资料"])
    assert result.source == "fallback"
    assert result.answerable is False
    assert "cache_only" in (result.error or "")


def test_missing_api_key_raises():
    client = ChatClient(provider="deepseek", api_key=None)
    client.api_key = None
    with pytest.raises(LLMError):
        client.chat("hi")
