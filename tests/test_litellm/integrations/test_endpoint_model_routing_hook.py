from typing import cast

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.endpoint_model_routing_hook import (
    EndpointModelRoutingHook,
    proxy_handler_instance,
)
from litellm.proxy._types import UserAPIKeyAuth


async def _run_hook(hook: EndpointModelRoutingHook, data: dict, call_type: str):
    return await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type=call_type,
    )


def test_proxy_handler_instance_is_endpoint_model_routing_hook():
    assert isinstance(proxy_handler_instance, EndpointModelRoutingHook)


@pytest.mark.asyncio
async def test_empty_lists_allow_all_models_and_call_types(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "")
    hook = EndpointModelRoutingHook()
    data = {"model": "any-model"}

    for call_type in ("anthropic_messages", "acompletion", "aresponses", "embedding"):
        assert await _run_hook(hook, data, call_type) is data


@pytest.mark.asyncio
async def test_anthropic_only_model_is_allowed_on_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "anthropic/*, claude-*")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "")
    hook = EndpointModelRoutingHook()
    data = {"model": "anthropic/a-1"}

    assert await _run_hook(hook, data, "anthropic_messages") is data


@pytest.mark.asyncio
async def test_anthropic_only_model_is_rejected_on_openai_chat(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "anthropic/*, claude-*")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "")
    hook = EndpointModelRoutingHook()

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, {"model": "anthropic/a-1"}, "acompletion")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Model 'anthropic/a-1' is only available through /v1/messages"


@pytest.mark.asyncio
async def test_openai_only_model_is_allowed_on_chat_and_responses(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "openai/*, gpt-*")
    hook = EndpointModelRoutingHook()
    data = {"model": "openai/o-1"}

    assert await _run_hook(hook, data, "acompletion") is data
    assert await _run_hook(hook, data, "aresponses") is data


@pytest.mark.asyncio
async def test_openai_only_model_is_rejected_on_anthropic_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "openai/*, gpt-*")
    hook = EndpointModelRoutingHook()

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, {"model": "openai/o-1"}, "anthropic_messages")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Model 'openai/o-1' is only available through OpenAI endpoints"


@pytest.mark.asyncio
async def test_rejection_displays_requested_alias_instead_of_resolved_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "openai/*")
    hook = EndpointModelRoutingHook()
    data = {
        "model": "openai/o-1",
        "proxy_server_request": {"body": {"model": "gpt-5.6-sol"}},
    }

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, data, "anthropic_messages")

    assert exc.value.detail == "Model 'gpt-5.6-sol' is only available through OpenAI endpoints"


@pytest.mark.asyncio
async def test_openai_only_model_is_rejected_on_non_openai_call_type(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "gpt-*")
    hook = EndpointModelRoutingHook()

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, {"model": "gpt-5"}, "embedding")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Model 'gpt-5' is only available through OpenAI endpoints"


@pytest.mark.asyncio
async def test_unmatched_models_are_not_restricted(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "anthropic/*")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "openai/*")
    hook = EndpointModelRoutingHook()
    data = {"model": "custom/a-1"}

    assert await _run_hook(hook, data, "anthropic_messages") is data
    assert await _run_hook(hook, data, "acompletion") is data


@pytest.mark.asyncio
async def test_matching_both_lists_is_rejected_on_every_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "shared-*")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "shared-*")
    hook = EndpointModelRoutingHook()

    for call_type in ("anthropic_messages", "acompletion"):
        with pytest.raises(HTTPException):
            await _run_hook(hook, {"model": "shared-model"}, call_type)
