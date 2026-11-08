import json
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter
from starlette.requests import Request
from starlette.types import Message

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.tamntlib.alias_aware_vision_model_router import AliasAwareVisionModelRouter
from litellm.integrations.tamntlib.model_client_restriction_hook import ModelClientRestrictionHook
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import (
    _resolve_router_settings_model_group_alias,  # pyright: ignore[reportPrivateUsage]  # exercise auth rewrite before hooks
)
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body,  # pyright: ignore[reportUnknownVariableType]  # legacy parser returns an unparameterized dict
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.litellm_pre_call_utils import (
    add_litellm_data_to_request,  # pyright: ignore[reportUnknownVariableType]  # validate legacy request output at the boundary
    refresh_proxy_server_request_body_snapshot,
)
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CallTypesLiteral

_POLICY: Final = "claude-*:claude-cli/2.1.308,Claude/3.6556.1;gpt-*:codex-tui/0.164.0,Codex Desktop/0.163.0"
_CLIENTS: Final = (
    ("claude-opus-4-7", "claude-cli"),
    ("claude-opus-4-7", "Claude"),
    ("gpt-test", "codex-tui"),
    ("gpt-test", "Codex Desktop"),
)


@pytest.fixture(autouse=True)
def policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_CLIENT_MIN_VERSIONS", raising=False)
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", _POLICY)


def _request_data(model: str = "claude-opus-4-7", user_agent: str | None = None) -> dict[str, object]:
    if user_agent is None:
        return {"model": model}
    return {"model": model, "proxy_server_request": {"headers": {"user-agent": user_agent}}}


async def _run_hook(data: dict[str, object]) -> dict[str, object]:
    return await ModelClientRestrictionHook().async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "acompletion")


async def _run_vision_hook(hook: CustomLogger, data: dict[str, object], call_type: CallTypesLiteral) -> object:
    return await hook.async_pre_call_hook(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # legacy callback dict
        UserAPIKeyAuth(), DualCache(), data, call_type
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_snapshot", [False, True])
@pytest.mark.parametrize("user_agent", ["curl/8.0", "codex-tui/0.164.0"])
async def test_auth_alias_rewrite_preserves_client_policy(
    monkeypatch: pytest.MonkeyPatch, user_agent: str, refresh_snapshot: bool
) -> None:
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", "gpt-*:codex-tui/0.164.0")
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {"model": "anthropic/primary", "api_key": "offline"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    token: Final = UserAPIKeyAuth(
        models=["anthropic/primary"], router_settings={"model_group_alias": {"gpt-*": "anthropic/primary"}}
    )
    data: Final[dict[str, object]] = {
        "model": "gpt-policy",
        "client_requested_model": "unrestricted",
        "metadata": {"client_requested_model": "unrestricted"},
        "litellm_metadata": {"client_requested_model": "unrestricted"},
        "proxy_server_request": {"client_requested_model": "unrestricted", "body": {"model": "unrestricted"}},
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": json.dumps(data).encode(), "more_body": False}

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"user-agent", user_agent.encode()), (b"content-type", b"application/json")],
        },
        receive=receive,
    )
    parsed: Final = TypeAdapter(dict[str, object]).validate_python(await read_request_body(request=request))
    assert parsed["model"] == "gpt-policy"
    await _resolve_router_settings_model_group_alias(data, token, request, "/v1/chat/completions")
    assert data["model"] == "anthropic/primary"
    forwarded: Final = TypeAdapter(dict[str, object]).validate_python(
        await add_litellm_data_to_request(
            data=data, request=request, user_api_key_dict=token, proxy_config=ProxyConfig(), general_settings={}
        )
    )
    snapshot: Final = TypeAdapter(dict[str, object]).validate_python(forwarded["proxy_server_request"])
    assert TypeAdapter(dict[str, object]).validate_python(snapshot["body"])["model"] == "anthropic/primary"
    if refresh_snapshot:
        refresh_proxy_server_request_body_snapshot(forwarded)
    if user_agent == "curl/8.0":
        with pytest.raises(HTTPException) as exc:
            await _run_hook(forwarded)
        assert exc.value.status_code == 403
        assert exc.value.detail == "Only configured clients are allowed for this model"
    else:
        assert await _run_hook(forwarded) is forwarded
    assert forwarded["model"] == "anthropic/primary"
    updated_snapshot: Final = TypeAdapter(dict[str, object]).validate_python(forwarded["proxy_server_request"])
    assert TypeAdapter(dict[str, object]).validate_python(updated_snapshot["body"])["model"] == "anthropic/primary"


@pytest.mark.asyncio
async def test_ingress_ignores_forged_original_model_without_auth_rewrite() -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-policy",
        "client_requested_model": "unrestricted",
        "metadata": {"client_requested_model": "unrestricted"},
        "litellm_metadata": {"client_requested_model": "unrestricted"},
        "proxy_server_request": {"client_requested_model": "unrestricted", "body": {"model": "unrestricted"}},
    }
    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"user-agent", b"curl/8.0")],
        }
    )
    forwarded: Final = TypeAdapter(dict[str, object]).validate_python(
        await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_config=ProxyConfig(),
            general_settings={},
        )
    )
    with pytest.raises(HTTPException) as exc:
        await _run_hook(forwarded)
    assert exc.value.status_code == 403
    assert forwarded["model"] == "gpt-policy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "call_type", "model", "allowed_agent", "stale_agent"),
    [
        ("/v1/responses", "aresponses", "gpt-policy", "codex-tui/0.164.0", "codex-tui/0.163.0"),
        ("/v1/chat/completions", "acompletion", "gpt-policy", "codex-tui/0.164.0", "codex-tui/0.163.0"),
        ("/v1/messages", "anthropic_messages", "claude-policy", "claude-cli/2.1.308", "claude-cli/2.1.307"),
    ],
)
@pytest.mark.parametrize("container", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize(
    ("agent_kind", "status"),
    [("stale", 400), ("missing", 403), ("empty", 403), ("whitespace", 403), ("other", 403), ("allowed", 200)],
)
async def test_ingress_headers_override_forged_user_agent_metadata(
    path: str,
    call_type: CallTypesLiteral,
    model: str,
    allowed_agent: str,
    stale_agent: str,
    container: str,
    agent_kind: str,
    status: int,
) -> None:
    user_agent: Final = {
        "stale": stale_agent,
        "missing": None,
        "empty": "",
        "whitespace": " \t ",
        "other": "curl/8.0",
        "allowed": allowed_agent,
    }[agent_kind]
    headers: Final = [] if user_agent is None else [(b"user-agent", user_agent.encode())]
    data: Final[dict[str, object]] = {
        "model": model,
        container: {"user_agent": "curl/8.0" if status == 200 else allowed_agent},
        "proxy_server_request": {"headers": {"user-agent": allowed_agent}},
    }
    request: Final = Request({"type": "http", "method": "POST", "path": path, "query_string": b"", "headers": headers})
    forwarded: Final = TypeAdapter(dict[str, object]).validate_python(
        await add_litellm_data_to_request(
            data=data,
            request=request,
            user_api_key_dict=UserAPIKeyAuth(),
            proxy_config=ProxyConfig(),
            general_settings={},
        )
    )
    snapshot: Final = TypeAdapter(dict[str, object]).validate_python(forwarded["proxy_server_request"])
    assert snapshot["headers"] == ({} if user_agent is None else {"user-agent": user_agent})
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    logging._init_litellm_callbacks()  # pyright: ignore[reportPrivateUsage]  # exercise proxy-registered policy hook
    hook: Final = logging.get_proxy_hook("model_client_restriction_hook")
    assert isinstance(hook, ModelClientRestrictionHook)
    if status == 200:
        assert await hook.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), forwarded, call_type) is forwarded
        return
    with pytest.raises(HTTPException) as exc:
        await hook.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), forwarded, call_type)
    assert exc.value.status_code == status
    client, minimum = allowed_agent.split("/")
    expected_detail: Final = (
        f"{client} version must be >= {minimum}. Please update {client} to the latest version."
        if status == 400
        else "Only configured clients are allowed for this model"
        if agent_kind == "other"
        else "Missing User-Agent header"
    )
    assert exc.value.detail == expected_detail


@pytest.mark.asyncio
async def test_generic_policy_enforces_codex_minimum() -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model="gpt-test", user_agent="codex-tui/0.163.0"))

    assert exc.value.status_code == 400
    assert "codex-tui version must be >= 0.164.0" in exc.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("public_model", "resolved_model", "user_agent", "status"),
    [
        ("gpt-policy", "openai/policy", "codex-tui/0.163.9", 400),
        ("gpt-policy", "openai/policy-vision", "curl/8.0", 403),
        ("claude-policy", "anthropic/policy", "claude-cli/2.1.307", 400),
        ("claude-policy", "anthropic/policy-vision", "curl/8.0", 403),
    ],
)
async def test_restricts_original_model_after_alias_resolution(
    public_model: str, resolved_model: str, user_agent: str, status: int
) -> None:
    data: Final[dict[str, object]] = {
        "model": resolved_model,
        "proxy_server_request": {"body": {"model": public_model}, "headers": {"user-agent": user_agent}},
    }

    with pytest.raises(HTTPException) as exc:
        await _run_hook(data)

    assert exc.value.status_code == status
    assert data["model"] == resolved_model


@pytest.fixture
def policy_router(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "openai/policy", "litellm_params": {"model": "openai/policy", "api_key": "offline"}},
            {"model_name": "anthropic/policy", "litellm_params": {"model": "anthropic/policy", "api_key": "offline"}},
        ],
        model_group_alias={"gpt-policy": "openai/policy", "claude-policy": "anthropic/policy"},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setenv("LITELLM_VISION_MODEL_ROUTER_MODELS", "openai/policy,anthropic/policy")


@pytest.mark.asyncio
@pytest.mark.usefixtures("policy_router")
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize(("client", "status"), [("old", 400), ("other", 403), ("valid", 200)])
@pytest.mark.parametrize(
    ("public_model", "resolved_model", "call_type", "old_agent", "valid_agent"),
    [
        ("gpt-policy", "openai/policy", "acompletion", "codex-tui/0.163.9", "codex-tui/0.164.0"),
        ("gpt-policy", "openai/policy", "aresponses", "codex-tui/0.163.9", "Codex Desktop/0.163.0-alpha.11.2"),
        ("claude-policy", "anthropic/policy", "anthropic_messages", "claude-cli/2.1.307", "claude-cli/2.1.308"),
    ],
)
async def test_vision_alias_chain_enforces_original_model_policy(
    public_model: str,
    resolved_model: str,
    call_type: CallTypesLiteral,
    old_agent: str,
    valid_agent: str,
    client: str,
    status: int,
    vision: bool,
) -> None:
    user_agent: Final = {"old": old_agent, "other": "curl/8.0", "valid": valid_agent}[client]
    image_type: Final = {"acompletion": "image_url", "aresponses": "input_image", "anthropic_messages": "image"}[
        call_type
    ]
    payload_key: Final = "input" if call_type == "aresponses" else "messages"
    content: Final = [{"type": image_type}] if vision else "hello"
    data: Final[dict[str, object]] = {
        "model": public_model,
        "proxy_server_request": {"body": {"model": public_model}, "headers": {"user-agent": user_agent}},
        payload_key: [{"role": "user", "content": content}],
    }
    routed: Final = await _run_vision_hook(AliasAwareVisionModelRouter(), data, call_type)
    forwarded: Final = TypeAdapter(dict[str, object]).validate_python(routed)
    routed_model: Final = resolved_model + ("-vision" if vision else "")
    assert forwarded["model"] == routed_model

    if status == 200:
        result: Final = await ModelClientRestrictionHook().async_pre_call_hook(
            UserAPIKeyAuth(), DualCache(), forwarded, call_type
        )
        assert result is forwarded
    else:
        with pytest.raises(HTTPException) as exc:
            await ModelClientRestrictionHook().async_pre_call_hook(UserAPIKeyAuth(), DualCache(), forwarded, call_type)
        assert exc.value.status_code == status

    assert forwarded["model"] == routed_model
    assert data["model"] == public_model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"body": None},
        {"body": []},
        {"body": {}},
        {"body": {"model": None}},
        {"body": {"model": ""}},
        {"body": {"model": 123}},
    ],
)
async def test_missing_snapshot_model_falls_back_to_current_model(snapshot: object) -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-policy",
        "metadata": {"user_agent": "curl/8.0"},
        "proxy_server_request": snapshot,
    }

    with pytest.raises(HTTPException) as exc:
        await _run_hook(data)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "user_agent"),
    [
        ("claude-opus-4-7", "claude-cli/2.1.308"),
        ("claude-sonnet-4-6", "Claude/3.6556.1"),
        ("claude-opus-4-7", "claude-cli/2.1.309-beta.1+build"),
        ("claude-opus-4-7", "Claude/3.6556.10"),
        ("claude-opus-4-7", "Claude/4.0.0"),
        ("gpt-test", "codex-tui/0.164.0"),
        ("gpt-test", "codex-tui/0.165.0"),
        ("gpt-test", "Codex Desktop/0.163.0"),
        ("gpt-test", "Codex Desktop/0.163.0-alpha.11.2"),
        ("gpt-test", "Codex Desktop/0.163.0-alpha.11.2+build.7"),
        ("gpt-test", "Codex Desktop/0.163.0+build.7"),
    ],
)
async def test_allows_each_client_at_or_above_its_numeric_minimum(model: str, user_agent: str) -> None:
    data: Final = _request_data(model=model, user_agent=user_agent)

    assert await _run_hook(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "user_agent"),
    [
        (
            "claude-opus-4-7",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Claude/3.6556.2 Chrome/152.0.7977.167 Electron/44.9.3 Safari/537.36",
        ),
        ("claude-opus-4-7", "proxy/1.0 CLAUDE-CLI/2.1.309 (external, cli) litellm/1.106.2"),
        ("claude-opus-4-7", "proxy/1.0 claude/3.6556.2"),
        ("gpt-test", "proxy/1.0 codex-tui/0.164.0 litellm/1.106.2"),
        ("gpt-test", "Codex Desktop/0.163.0-alpha.11.2 (Mac OS 26.6.2; arm64) unknown (Codex Desktop)"),
        ("gpt-test", "proxy/1.0 CODEX DESKTOP/0.163.0-alpha.11.2 litellm/1.106.2"),
    ],
)
async def test_reads_product_tokens_in_full_user_agents(model: str, user_agent: str) -> None:
    data: Final = _request_data(model=model, user_agent=user_agent)

    assert await _run_hook(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "client", "version", "minimum"),
    [
        ("claude-opus-4-7", "claude-cli", "2.1.307", "2.1.308"),
        ("claude-opus-4-7", "Claude", "3.6556.0", "3.6556.1"),
        ("gpt-test", "codex-tui", "0.163.9", "0.164.0"),
        ("gpt-test", "Codex Desktop", "0.162.9-alpha.11.2", "0.163.0"),
    ],
)
async def test_rejects_old_versions_in_full_user_agents(model: str, client: str, version: str, minimum: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model=model, user_agent=f"proxy/1.0 {client}/{version} litellm/1"))

    assert exc.value.status_code == 400
    assert exc.value.detail == f"{client} version must be >= {minimum}. Please update {client} to the latest version."


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "client"), _CLIENTS)
@pytest.mark.parametrize("prefix", ["Not", "proxy/", "proxy/1.0 X-"])
async def test_rejects_embedded_client_name_fragments(model: str, client: str, prefix: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model=model, user_agent=f"{prefix}{client}/9.9.9"))

    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "user_agent"),
    [
        ("claude-opus-4-7", "codex-tui/0.164.0"),
        ("claude-opus-4-7", "Codex Desktop/0.163.0-alpha.11.2"),
        ("gpt-test", "claude-cli/2.1.308"),
        ("gpt-test", "Claude/3.6556.1"),
        ("gpt-test", "codex_cli_rs/0.164.0"),
        ("gpt-test", "curl/8.0"),
    ],
)
async def test_rejects_clients_not_allowed_by_the_matching_rule(model: str, user_agent: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model=model, user_agent=user_agent))

    assert exc.value.status_code == 403
    assert exc.value.detail == "Only configured clients are allowed for this model"


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "client"), _CLIENTS)
@pytest.mark.parametrize("prefix", ["", "proxy/1.0 "])
@pytest.mark.parametrize(
    "version",
    [
        "",
        " \t",
        " 9.9.9",
        "not-a-version",
        "1.2",
        "1..3",
        "1.2.3.4",
        "-beta",
        "+build",
        pytest.param("².2.3", id="non-decimal-digit"),
        pytest.param("١.٢.٣", id="non-ascii-digits"),
        pytest.param("9" * 5000 + ".2.3", id="oversized-integer"),
    ],
)
async def test_rejects_malformed_versions(model: str, client: str, prefix: str, version: str) -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model=model, user_agent=f"{prefix}{client}/{version}"))

    assert exc.value.status_code == 400
    assert exc.value.detail == f"Invalid {client} version"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_agent", [None, "", " \t "])
async def test_rejects_missing_user_agent(user_agent: str | None) -> None:
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(user_agent=user_agent))

    assert exc.value.status_code == 403
    assert exc.value.detail == "Missing User-Agent header"


@pytest.mark.asyncio
@pytest.mark.parametrize("container", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("snapshot", [None, {}, {"headers": None}, {"headers": []}, {"headers": "invalid"}])
async def test_reads_user_agent_from_metadata(container: str, snapshot: object) -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-test",
        container: {"user_agent": "codex-tui/0.164.0"},
        "proxy_server_request": snapshot,
    }

    assert await _run_hook(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["user-agent", "User-Agent"])
async def test_reads_user_agent_from_headers(header: str) -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-test",
        "proxy_server_request": {"headers": {header: "Codex Desktop/0.163.0-alpha.11.2"}},
    }

    assert await _run_hook(data) is data


@pytest.mark.asyncio
async def test_metadata_takes_precedence_over_litellm_metadata_without_header_snapshot() -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-test",
        "metadata": {"user_agent": "curl/8.0"},
        "litellm_metadata": {"user_agent": "codex-tui/0.164.0"},
    }

    with pytest.raises(HTTPException) as exc:
        await _run_hook(data)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Only configured clients are allowed for this model"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_agent", [None, "curl/8.0", "claude-cli/", "Codex Desktop/not-a-version"])
async def test_skips_models_without_a_matching_rule(user_agent: str | None) -> None:
    data: Final = _request_data(model="unrestricted-model", user_agent=user_agent)

    assert await _run_hook(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "", "  ", " ; ; "])
async def test_empty_policy_is_disabled(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("MODEL_CLIENT_RESTRICTIONS", raising=False)
    else:
        monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", value)
    data: Final = _request_data(user_agent="curl/8.0")

    assert await _run_hook(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        "claude-*",
        ":Claude/1.2.3",
        "claude-*:",
        "claude-*:Claude",
        "claude-*:Claude/",
        "claude-*:Claude/1.2",
        "claude-*:Claude/1..3",
        "claude-*:Claude/1.2.3.4",
        "claude-*:Claude/-1.2.3",
        "claude-*:Claude/1.2.3-alpha.1",
        "claude-*:Claude/1.2.3+build",
        "claude-*:Claude/².2.3",
        "claude-*:Claude/١.٢.٣",
        "claude-*:Claude/1.2.3,",
        "claude-*:proxy/Claude/1.2.3",
        "claude-*:Claude/1.2.3;broken-rule",
        pytest.param("claude-*:Claude/" + "9" * 5000 + ".2.3", id="oversized-integer"),
    ],
)
async def test_invalid_policy_fails_closed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", value)

    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(user_agent="Claude/9.9.9"))

    assert exc.value.status_code == 500
    assert "Invalid MODEL_CLIENT_RESTRICTIONS" in exc.value.detail


@pytest.mark.asyncio
async def test_arbitrary_client_names_are_literal_and_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", "custom-*:My.Agent+/1.2.3,codex_cli_rs/0.164.0")
    data: Final = _request_data(model="custom-test", user_agent="proxy/1.0 My.Agent+/1.2.3")
    other_data: Final = _request_data(model="custom-test", user_agent="codex_cli_rs/0.164.0")

    assert await _run_hook(data) is data
    assert await _run_hook(other_data) is other_data
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model="custom-test", user_agent="MyXAgent/1.2.3"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_first_matching_model_rule_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", "gpt-*:codex-tui/0.164.0;gpt-special:Claude/3.6556.1")
    data: Final = _request_data(model="gpt-special", user_agent="codex-tui/0.164.0")

    assert await _run_hook(data) is data
    with pytest.raises(HTTPException) as exc:
        await _run_hook(_request_data(model="gpt-special", user_agent="Claude/3.6556.1"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_whitespace_and_colons_in_model_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CLIENT_RESTRICTIONS", " provider:model-* : claude-cli/2.1.308 , Claude/3.6556.1 ; ")
    data: Final = _request_data(model="provider:model-test", user_agent="Claude/3.6556.1")

    assert await _run_hook(data) is data


@pytest.mark.asyncio
async def test_retired_claude_environment_does_not_enable_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_CLIENT_RESTRICTIONS", raising=False)
    monkeypatch.setenv("MODEL_CLIENT_MIN_VERSIONS", _POLICY)
    monkeypatch.setenv("CLAUDE_CLIENT_MODELS", "claude-*")
    monkeypatch.setenv("CLAUDE_CODE_MODELS", "claude-*")
    monkeypatch.setenv("CLAUDE_CODE_MIN_VERSION", "2.1.309")
    monkeypatch.setenv("CLAUDE_DESKTOP_MIN_VERSION", "3.6556.2")
    data: Final = _request_data(user_agent="curl/8.0")

    assert await _run_hook(data) is data
