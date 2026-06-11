from typing import cast

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.alias_aware_vision_model_router import (
    AliasAwareVisionModelRouter,
    has_vision,
)
from litellm.proxy._types import UserAPIKeyAuth

VISION_MODELS_ENV_VAR = "LITELLM_VISION_MODEL_ROUTER_MODELS"


def _make_image_data(image_block: dict, model: str = "anthropic/primary") -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    image_block,
                ],
            }
        ],
    }


def _set_alias_map(monkeypatch, alias_map: dict) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", alias_map)


@pytest.mark.asyncio
async def test_noops_when_env_is_empty(monkeypatch):
    monkeypatch.delenv(VISION_MODELS_ENV_VAR, raising=False)
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_noops_when_model_not_configured(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data(
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        model="openai/primary",
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    assert data["model"] == "openai/primary"


@pytest.mark.asyncio
async def test_noops_for_text_only_configured_model(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = {"model": "anthropic/primary", "messages": [{"role": "user", "content": "hello"}]}

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_text_only_alias_resolves_to_configured_model(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {"gpta-1": "anthropic/primary"})
    hook = AliasAwareVisionModelRouter()
    data = {"model": "gpta-1", "messages": [{"role": "user", "content": "hello"}]}

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary"
    assert data["model"] == "gpta-1"


@pytest.mark.asyncio
async def test_routes_direct_openai_chat_image_to_vision_suffix(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"
    assert cast(dict, result)["messages"] is data["messages"]
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_routes_litellm_alias_to_resolved_model_vision_suffix(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {"gpta-1": "anthropic/primary"})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data(
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        model="gpta-1",
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"
    assert data["model"] == "gpta-1"


@pytest.mark.asyncio
async def test_routes_router_alias_to_resolved_model_vision_suffix(monkeypatch, mocker):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    mocker.patch(
        "litellm.integrations.alias_aware_vision_model_router._resolve_router_model_alias",
        return_value="anthropic/primary",
    )
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data(
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        model="gpta-1",
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"
    assert data["model"] == "gpta-1"


@pytest.mark.asyncio
async def test_routes_openai_responses_image_to_vision_suffix(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = {
        "model": "anthropic/primary",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                ],
            }
        ],
    }

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="aresponses",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"
    assert cast(dict, result)["input"] is data["input"]


@pytest.mark.asyncio
async def test_routes_anthropic_image_to_vision_suffix(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        }
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="anthropic_messages",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"


@pytest.mark.asyncio
async def test_routes_anthropic_tool_result_image_to_vision_suffix(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = {
        "model": "anthropic/primary",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_iF8ftgbP1x9qI5jr3RPtcXbt",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgo",
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="anthropic_messages",
    )

    assert result is not None
    assert cast(dict, result)["model"] == "anthropic/primary-vision"


@pytest.mark.asyncio
async def test_openai_chat_ignores_responses_image_blocks(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data({"type": "input_image", "image_url": "https://example.com/cat.png"})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_openai_responses_ignores_chat_image_blocks(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = {
        "model": "anthropic/primary",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
    }

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="aresponses",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_openai_call_type_ignores_anthropic_image_blocks(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        }
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_anthropic_call_type_ignores_openai_image_blocks(monkeypatch):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="anthropic_messages",
    )

    assert result is None
    assert data["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_logs_and_noops_when_vision_detection_raises(monkeypatch, mocker):
    monkeypatch.setenv(VISION_MODELS_ENV_VAR, "anthropic/primary")
    _set_alias_map(monkeypatch, {})
    mocker.patch(
        "litellm.integrations.alias_aware_vision_model_router.has_vision",
        side_effect=Exception("boom"),
    )
    exception_log = mocker.patch("litellm.integrations.alias_aware_vision_model_router.verbose_logger.exception")
    hook = AliasAwareVisionModelRouter()
    data = _make_image_data({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type="acompletion",
    )

    assert result is None
    exception_log.assert_called_once_with("AliasAwareVisionModelRouter: failed to route model")


def test_has_vision_returns_false_for_unknown_call_type():
    data = _make_image_data({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})

    assert has_vision(data, "unknown") is False


def test_proxy_handler_instance_is_alias_aware_vision_model_router():
    from litellm.integrations.alias_aware_vision_model_router import proxy_handler_instance

    assert isinstance(proxy_handler_instance, AliasAwareVisionModelRouter)
