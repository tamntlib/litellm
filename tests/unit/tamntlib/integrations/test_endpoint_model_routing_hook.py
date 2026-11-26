import json
from typing import Final, cast

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter
from starlette.requests import Request
from starlette.types import Message

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.integrations.tamntlib.endpoint_model_routing_hook import (
    EndpointModelRoutingHook,
    proxy_handler_instance,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import (
    _resolve_router_settings_model_group_alias,  # pyright: ignore[reportPrivateUsage]  # exercise auth before registered hooks
)
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body,  # pyright: ignore[reportUnknownVariableType]  # legacy parser returns an unparameterized dict
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.litellm_pre_call_utils import (
    add_litellm_data_to_request,  # pyright: ignore[reportUnknownVariableType]  # validate the legacy request output below
)
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CallTypesLiteral


@pytest.mark.asyncio
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses", "anthropic_messages"])
@pytest.mark.parametrize("family", ["anthropic", "openai"])
@pytest.mark.parametrize(
    ("alias_stage", "policy_identity"),
    [
        ("direct", "base"),
        ("router", "public"),
        ("router", "base"),
        ("auth", "public"),
        ("auth", "prevision"),
        ("auth", "base"),
        ("auth", "routed"),
        ("auth", "routed-target"),
    ],
)
async def test_registered_vision_chain_preserves_endpoint_restrictions(
    monkeypatch: pytest.MonkeyPatch,
    alias_stage: str,
    policy_identity: str,
    family: str,
    call_type: CallTypesLiteral,
    vision: bool,
) -> None:
    base: Final = f"{family}/primary"
    routed: Final = f"{base}-vision"
    vision_target: Final = f"{family}/image-deployment"
    public: Final = base if alias_stage == "direct" else "client-a"
    prevision: Final = "base-a" if alias_stage == "auth" else public
    pattern: Final = {
        "public": "client-*",
        "prevision": prevision,
        "base": base,
        "routed": routed,
        "routed-target": vision_target,
    }[policy_identity]
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", pattern if family == "anthropic" else "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", pattern if family == "openai" else "")
    monkeypatch.setenv("LITELLM_VISION_MODEL_ROUTER_MODELS", base)
    monkeypatch.delenv("MODEL_CLIENT_RESTRICTIONS", raising=False)
    router: Final = Router(
        model_list=[
            {"model_name": model, "litellm_params": {"model": model, "api_key": "offline"}}
            for model in (base, vision_target)
        ],
        model_group_alias={"client-*": base, "base-*": base, routed: vision_target},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    token: Final = UserAPIKeyAuth(
        router_settings={"model_group_alias": {"client-*": prevision}} if alias_stage == "auth" else {}
    )
    path: Final = {
        "acompletion": "/v1/chat/completions",
        "aresponses": "/v1/responses",
        "anthropic_messages": "/v1/messages",
    }[call_type]
    image_type: Final = {"acompletion": "image_url", "aresponses": "input_image", "anthropic_messages": "image"}[
        call_type
    ]
    data: Final[dict[str, object]] = {
        "model": public,
        "input" if call_type == "aresponses" else "messages": [
            {"role": "user", "content": [{"type": image_type}] if vision else "hello"}
        ],
        "proxy_server_request": {"client_requested_model": "unrestricted", "body": {"model": "unrestricted"}},
    }
    body: Final = json.dumps(data).encode()

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request: Final = Request(
        {"type": "http", "method": "POST", "path": path, "query_string": b"", "headers": []}, receive=receive
    )
    assert (await read_request_body(request=request))["model"] == public
    if alias_stage == "auth":
        await _resolve_router_settings_model_group_alias(data, token, request, path)
    assert data["model"] == prevision
    forwarded: Final = TypeAdapter(dict[str, object]).validate_python(
        await add_litellm_data_to_request(
            data=data, request=request, user_api_key_dict=token, proxy_config=ProxyConfig(), general_settings={}
        )
    )
    snapshot: Final = TypeAdapter(dict[str, object]).validate_python(forwarded["proxy_server_request"])
    snapshot_body: Final = TypeAdapter(dict[str, object]).validate_python(snapshot["body"])
    assert snapshot["client_requested_model"] == (public if alias_stage == "auth" else None)
    assert snapshot_body["model"] == prevision
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    logging._init_litellm_callbacks()  # pyright: ignore[reportPrivateUsage]  # use the real mandatory callback registration
    permitted: Final = call_type == "anthropic_messages" if family == "anthropic" else call_type != "anthropic_messages"
    restricted: Final = vision or policy_identity not in ("routed", "routed-target")

    if restricted and not permitted:
        with pytest.raises(HTTPException) as exc:
            await logging.pre_call_hook(  # pyright: ignore[reportUnknownMemberType]  # legacy callback output is discarded on rejection
                user_api_key_dict=token, data=forwarded, call_type=call_type
            )
        assert exc.value.status_code == 400
        endpoint: Final = "/v1/messages" if family == "anthropic" else "OpenAI endpoints"
        assert exc.value.detail == f"Model '{public}' is only available through {endpoint}"
    else:
        result: Final = TypeAdapter(dict[str, object]).validate_python(
            await logging.pre_call_hook(  # pyright: ignore[reportUnknownMemberType]  # validate legacy callback output at the boundary
                user_api_key_dict=token, data=forwarded, call_type=call_type
            )
        )
        expected: Final = routed if vision else base
        assert result["model"] == expected
        deployment: Final = TypeAdapter(dict[str, object]).validate_python(
            router.get_available_deployment(model=expected)  # pyright: ignore[reportUnknownMemberType]  # legacy deployment output
        )
        assert deployment["model_name"] == (vision_target if vision else base)
        assert await _run_hook(EndpointModelRoutingHook(), result, call_type) is result
        assert result["model"] == expected
    assert forwarded["model"] == prevision
    updated_snapshot: Final = TypeAdapter(dict[str, object]).validate_python(forwarded["proxy_server_request"])
    assert TypeAdapter(dict[str, object]).validate_python(updated_snapshot["body"])["model"] == prevision
    assert updated_snapshot["client_requested_model"] == (public if alias_stage == "auth" else None)


async def _run_hook(hook: EndpointModelRoutingHook, data: dict, call_type: str):
    return await hook.async_pre_call_hook(
        user_api_key_dict=cast(UserAPIKeyAuth, None),
        cache=cast(DualCache, None),
        data=data,
        call_type=call_type,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_pattern", ["client-a", "client-*"])
@pytest.mark.parametrize("policy_pattern", ["target", "target-wildcard", "public-only"])
@pytest.mark.parametrize("family", ["anthropic", "openai"])
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses", "anthropic_messages", "embedding"])
async def test_router_alias_policy_without_vision_configuration(
    monkeypatch: pytest.MonkeyPatch, alias_pattern: str, policy_pattern: str, family: str, call_type: str
) -> None:
    monkeypatch.delenv("LITELLM_VISION_MODEL_ROUTER_MODELS", raising=False)
    target: Final = f"{family}/primary"
    router: Final = Router(
        model_list=[{"model_name": target, "litellm_params": {"model": target, "api_key": "offline"}}],
        model_group_alias={alias_pattern: target},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    pattern: Final = {"target": target, "target-wildcard": f"{family}/*", "public-only": "client-*"}[policy_pattern]
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", pattern if family == "anthropic" else "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", pattern if family == "openai" else "")
    data: Final = {"model": "client-a"}
    permitted: Final = (
        call_type == "anthropic_messages" if family == "anthropic" else call_type in ("acompletion", "aresponses")
    )

    if permitted:
        assert await _run_hook(EndpointModelRoutingHook(), data, call_type) is data
    else:
        with pytest.raises(HTTPException) as exc:
            await _run_hook(EndpointModelRoutingHook(), data, call_type)
        assert exc.value.status_code == 400
        endpoint: Final = "/v1/messages" if family == "anthropic" else "OpenAI endpoints"
        assert exc.value.detail == f"Model 'client-a' is only available through {endpoint}"
    assert data["model"] == "client-a"


@pytest.mark.asyncio
async def test_router_alias_is_unrestricted_with_empty_endpoint_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_VISION_MODEL_ROUTER_MODELS", raising=False)
    monkeypatch.setenv("ANTHROPIC_ONLY_MODELS", "")
    monkeypatch.setenv("OPENAI_ONLY_MODELS", "")
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {"model": "anthropic/primary", "api_key": "offline"},
            }
        ],
        model_group_alias={"client-*": "anthropic/primary"},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    data: Final = {"model": "client-a"}
    for call_type in ("acompletion", "aresponses", "anthropic_messages", "embedding"):
        assert await _run_hook(EndpointModelRoutingHook(), data, call_type) is data
    assert data["model"] == "client-a"


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
