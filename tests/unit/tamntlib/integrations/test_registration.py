import importlib
import importlib.util
from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging


@pytest.mark.asyncio
async def test_package_hooks_initialize_once_and_preserve_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    assert importlib.util.find_spec("litellm.integrations.tamntlib") is not None
    package: Final = importlib.import_module("litellm.integrations.tamntlib")
    assert package.__path__
    hook_types: Final = {
        "alias_aware_vision_model_router": ("alias_aware_vision_model_router", "AliasAwareVisionModelRouter"),
        "claude_code_version_check_hook": ("claude_code_version_check_hook", "ClaudeCodeVersionCheckHook"),
        "endpoint_model_routing": ("endpoint_model_routing_hook", "EndpointModelRoutingHook"),
        "client_user_agent": ("client_user_agent_hook", "ClientUserAgentHook"),
    }
    monkeypatch.delenv("LITELLM_FORWARD_CLIENT_USER_AGENT", raising=False)
    monkeypatch.delenv("LITELLM_VISION_MODEL_ROUTER_MODELS", raising=False)
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    logging._init_litellm_callbacks()
    for name, (module_name, class_name) in hook_types.items():
        module: Final = importlib.import_module(f"litellm.integrations.tamntlib.{module_name}")
        hook_type: Final = getattr(module, class_name)
        instances: Final = tuple(callback for callback in litellm.callbacks if isinstance(callback, hook_type))
        assert len(instances) == 1, name
        assert logging.get_proxy_hook(name) is instances[0]
    data: Final = {
        "model": "anthropic/primary",
        "proxy_server_request": {"headers": {"user-agent": "client/1"}},
        "extra_headers": {"User-Agent": "configured/1"},
    }
    from litellm.types.utils import CallTypes

    ua_hook: Final = logging.get_proxy_hook("client_user_agent")
    assert ua_hook is not None
    assert await ua_hook.async_pre_call_deployment_hook(data, CallTypes.acompletion) is None
    assert data["extra_headers"] == {"User-Agent": "configured/1"}
    vision_hook: Final = logging.get_proxy_hook("alias_aware_vision_model_router")
    assert vision_hook is not None
    assert await vision_hook.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "acompletion") is None
    assert data["model"] == "anthropic/primary"
