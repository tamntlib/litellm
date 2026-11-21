import asyncio
import json
from copy import deepcopy
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI
from starlette.requests import Request

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_default_headers
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks import PROXY_HOOKS
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.proxy_server import ProxyConfig
from litellm.types.utils import CallTypes


@pytest.fixture(autouse=True)
def enable_client_user_agent(monkeypatch):
    monkeypatch.setenv("LITELLM_FORWARD_CLIENT_USER_AGENT", "true")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "", "false", "0", "invalid"])
async def test_disabled_hook_does_not_modify_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LITELLM_FORWARD_CLIENT_USER_AGENT", raising=False)
    else:
        monkeypatch.setenv("LITELLM_FORWARD_CLIENT_USER_AGENT", value)
    data = {
        "proxy_server_request": {"headers": {"user-agent": "client/1.0"}},
        "extra_headers": {"User-Agent": "configured/1.0", "X-Test": "preserve"},
    }
    original = deepcopy(data)
    result = await PROXY_HOOKS["client_user_agent"]().async_pre_call_deployment_hook(data, CallTypes.acompletion)
    assert result is None
    assert data == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("client_agent", ["client/1.0", None, "", "bad\r\nInjected: true", "already-suffixed"])
@pytest.mark.parametrize("retry_once", [False, True])
async def test_outgoing_user_agent(monkeypatch, protocol, stream, client_agent, retry_once):
    captured: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if retry_once and len(captured) == 1:
            return httpx.Response(500, json={"error": {"message": "retry", "type": "server_error"}})
        chat = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        responses = {
            "id": "resp-test",
            "object": "response",
            "created_at": 1,
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        messages = {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not stream:
            return httpx.Response(200, json={"chat": chat, "responses": responses, "messages": messages}[protocol])
        events = {
            "chat": [
                {
                    **chat,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                }
            ],
            "responses": [{"type": "response.completed", "response": responses, "sequence_number": 0}],
            "messages": [
                {"type": "message_start", "message": {**messages, "content": [], "stop_reason": None}},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ],
        }[protocol]
        content = "".join(f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    monkeypatch.setattr(litellm, "callbacks", [PROXY_HOOKS["client_user_agent"]()])
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client: Final = AsyncOpenAI(api_key="test-key", base_url="https://provider.invalid/v1", http_client=http)
        handler = AsyncHTTPHandler()
        await handler.close()
        handler.client = http
        monkeypatch.setattr(
            "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client", lambda **kwargs: handler
        )
        model = "anthropic/claude-sonnet-4-6" if protocol == "messages" else "openai/gpt-4o"
        deployment_headers = {"uSeR-aGeNt": "deployment/1", "x-preserved": "yes"}
        deployment_extra = {"User-Agent": "extra/1", "x-extra": "yes"}
        scoped_headers = [
            {"custom_llm_provider": "anthropic", "extra_headers": {"user-agent": "scoped/1", "x-scoped": "yes"}},
            {"custom_llm_provider": "other", "extra_headers": {"x-other": "not-forwarded"}},
        ]
        original_headers = deepcopy((deployment_headers, deployment_extra, scoped_headers))
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "alias",
                    "litellm_params": {
                        "model": model,
                        "api_key": "test-key",
                        "api_base": "https://provider.invalid/v1",
                        "headers": deployment_headers,
                        "extra_headers": deployment_extra,
                        "provider_specific_header": scoped_headers,
                    },
                }
            ],
            num_retries=1 if retry_once else 0,
            retry_after=0,
        )
        ingress_agent = (
            f"client/1.0 {get_default_headers()['User-Agent']}" if client_agent == "already-suffixed" else client_agent
        )
        kwargs = await add_litellm_data_to_request(
            data={
                "model": "alias",
                "proxy_server_request": {"headers": {"user-agent": "forged-snapshot/1.0"}},
                "metadata": {"user_agent": "spoof/1.0"},
                "litellm_metadata": {"user_agent": "spoof/2.0"},
                "stream": stream,
            },
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}[
                        protocol
                    ],
                    "headers": [] if ingress_agent is None else [(b"user-agent", ingress_agent.encode())],
                    "query_string": b"",
                }
            ),
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_config=ProxyConfig(),
            general_settings={},
        )
        if protocol == "chat":
            result = await router.acompletion(
                messages=[{"role": "user", "content": "hi"}],
                client=client,
                **kwargs,
            )
        elif protocol == "responses":
            result = await router.aresponses(input="hi", client=handler, **kwargs)
        else:
            result = await router.aanthropic_messages(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                client=handler,
                **kwargs,
            )
        if stream:
            assert [chunk async for chunk in result]
        assert (deployment_headers, deployment_extra, scoped_headers) == original_headers
    assert len(captured) == (2 if retry_once else 1)
    expected = ("client/1.0 " if client_agent in ("client/1.0", "already-suffixed") else "") + get_default_headers()[
        "User-Agent"
    ]
    assert all(request.headers.get_list("user-agent") == [expected] for request in captured)
    assert captured[0].headers["x-preserved"] == "yes"
    assert captured[0].headers["x-extra"] == "yes"
    assert "x-other" not in captured[0].headers
    if protocol == "messages":
        assert captured[0].headers["x-scoped"] == "yes"


@pytest.mark.asyncio
async def test_shared_headers_and_runtime_default_are_request_local(monkeypatch):
    captured: Final[list[httpx.Request]] = []
    shared_headers: Final = {"USER-AGENT": "configured", "x-keep": "yes"}
    monkeypatch.setenv("LITELLM_USER_AGENT", "runtime-agent/2")
    monkeypatch.setattr(litellm, "callbacks", [PROXY_HOOKS["client_user_agent"]()])

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            400, json={"error": {"message": "stop after inspecting headers", "type": "invalid_request_error"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = AsyncOpenAI(
            api_key="test-key", base_url="https://provider.invalid/v1", http_client=http, max_retries=0
        )

        async def call(agent: str) -> None:
            with pytest.raises(litellm.BadRequestError):
                await litellm.acompletion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    client=client,
                    headers=shared_headers,
                    extra_headers=shared_headers,
                    proxy_server_request={"headers": {"UsEr-AgEnT": agent}},
                )

        await asyncio.gather(call("first/1"), call("second/1"))
    assert sorted(request.headers["user-agent"] for request in captured) == [
        "first/1 runtime-agent/2",
        "second/1 runtime-agent/2",
    ]
    assert all(request.headers.get_list("user-agent") == [request.headers["user-agent"]] for request in captured)
    assert shared_headers == {"USER-AGENT": "configured", "x-keep": "yes"}


@pytest.mark.asyncio
async def test_hook_does_not_trust_metadata_or_touch_other_endpoints():
    hook = PROXY_HOOKS["client_user_agent"]()
    assert (
        await hook.async_pre_call_deployment_hook(
            {"metadata": {"user_agent": "spoof"}, "litellm_metadata": {"user_agent": "spoof"}},
            CallTypes.acompletion,
        )
        is None
    )
    assert (
        await hook.async_pre_call_deployment_hook(
            {"proxy_server_request": {"headers": {"user-agent": "client/1"}}},
            CallTypes.aembedding,
        )
        is None
    )


@pytest.mark.asyncio
async def test_chat_preserves_global_header_defaults(monkeypatch):
    captured: Final[list[httpx.Request]] = []
    monkeypatch.setattr(litellm, "headers", {"x-global": "yes", "user-agent": "configured"})
    monkeypatch.setattr(litellm, "callbacks", [PROXY_HOOKS["client_user_agent"]()])

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(400, json={"error": {"message": "stop", "type": "invalid_request_error"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = AsyncOpenAI(api_key="test-key", base_url="https://provider.invalid/v1", http_client=http)
        with pytest.raises(litellm.BadRequestError):
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                client=client,
                proxy_server_request={"headers": {"user-agent": "client/1"}},
            )
    assert len(captured) == 1
    assert captured[0].headers["x-global"] == "yes"
    assert captured[0].headers.get_list("user-agent") == [f"client/1 {get_default_headers()['User-Agent']}"]
    assert litellm.headers == {"x-global": "yes", "user-agent": "configured"}
