import pytest
from fastapi import HTTPException

from litellm.integrations.tamntlib.claude_code_version_check_hook import (
    ClaudeCodeVersionCheckHook,
    proxy_handler_instance,
)


def _request_data(model: str = "claude-opus-4-7", user_agent: str | None = None):
    data: dict = {"model": model}
    if user_agent is not None:
        data["proxy_server_request"] = {"headers": {"user-agent": user_agent}}
    return data


async def _run_hook(hook: ClaudeCodeVersionCheckHook, data: dict):
    return await hook.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="acompletion",
    )


@pytest.fixture
def hook(monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_CODE_MODELS",
        "claude-opus-4-7,claude-sonnet-4-6",
    )
    monkeypatch.setenv("CLAUDE_CODE_MIN_VERSION", "1.2.3")
    return ClaudeCodeVersionCheckHook()


def test_proxy_handler_instance_is_claude_code_version_check_hook():
    assert isinstance(proxy_handler_instance, ClaudeCodeVersionCheckHook)


@pytest.mark.asyncio
async def test_allows_configured_model_with_minimum_claude_code_version(hook):
    data = _request_data(user_agent="claude-cli/1.2.3")

    result = await _run_hook(hook, data)

    assert result is data


@pytest.mark.asyncio
async def test_allows_prerelease_or_build_metadata_when_numeric_version_matches(hook):
    data = _request_data(user_agent="claude-cli/1.2.3-beta.1+build")

    result = await _run_hook(hook, data)

    assert result is data


@pytest.mark.asyncio
async def test_reads_user_agent_from_metadata(hook):
    data = {
        "model": "claude-opus-4-7",
        "metadata": {"user_agent": "claude-cli/1.2.4"},
    }

    result = await _run_hook(hook, data)

    assert result is data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_agent", [None, "curl/8.0", "claude-cli/", "claude-cli/ \t", "claude-cli/not-a-version"]
)
async def test_skips_unconfigured_model_regardless_of_user_agent(hook, user_agent):
    data = _request_data(model="gpt-4o", user_agent=user_agent)

    result = await _run_hook(hook, data)

    assert result is data


@pytest.mark.asyncio
async def test_rejects_missing_user_agent_for_configured_model(hook):
    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, _request_data())

    assert exc.value.status_code == 403
    assert exc.value.detail == "Missing User-Agent header"


@pytest.mark.asyncio
async def test_rejects_non_claude_code_user_agent(hook):
    data = _request_data(user_agent="curl/8.0")

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, data)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Only Claude Code is allowed"


@pytest.mark.asyncio
async def test_rejects_claude_code_below_minimum_version(hook):
    data = _request_data(user_agent="claude-cli/1.2.2")

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, data)

    assert exc.value.status_code == 400
    assert (
        exc.value.detail
        == "Claude Code version must be >= 1.2.3. Please update Claude Code to the latest version."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    [
        "not-a-version",
        "",
        " \t",
        "1.2",
        "1..3",
        "1.2.3.4",
        "-beta",
        "+build",
        pytest.param("².2.3", id="non-decimal-digit"),
        pytest.param("9" * 5000 + ".2.3", id="oversized-integer"),
    ],
)
async def test_rejects_invalid_client_version(hook, version):
    data = _request_data(user_agent=f"claude-cli/{version}")

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, data)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid Claude Code version"


@pytest.mark.asyncio
async def test_wildcard_pattern_enforces_matching_model(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MODELS", "claude-*")
    monkeypatch.setenv("CLAUDE_CODE_MIN_VERSION", "1.2.3")
    hook = ClaudeCodeVersionCheckHook()

    with pytest.raises(HTTPException) as exc:
        await _run_hook(hook, _request_data(model="claude-sonnet-4-6"))

    assert exc.value.status_code == 403
    assert exc.value.detail == "Missing User-Agent header"


@pytest.mark.asyncio
async def test_wildcard_pattern_skips_non_matching_model(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MODELS", "claude-*")
    monkeypatch.setenv("CLAUDE_CODE_MIN_VERSION", "1.2.3")
    hook = ClaudeCodeVersionCheckHook()
    data = _request_data(model="gpt-4o")

    result = await _run_hook(hook, data)

    assert result is data


@pytest.mark.asyncio
async def test_rejects_invalid_minimum_version(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MODELS", "claude-opus-4-7")
    monkeypatch.setenv("CLAUDE_CODE_MIN_VERSION", "1.2")
    data = _request_data(user_agent="claude-cli/1.2.3")

    with pytest.raises(HTTPException) as exc:
        await _run_hook(ClaudeCodeVersionCheckHook(), data)

    assert exc.value.status_code == 500
    assert exc.value.detail == "Invalid CLAUDE_CODE_MIN_VERSION env var: 1.2"
