import json

import pytest
from starlette.requests import Request

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import _can_object_call_model
from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.common_utils import resolve_model_group_alias


def test_can_object_call_model_access_via_wildcard_alias_underlying_model():
    from litellm import Router
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    llm_router = Router(
        model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {
                    "model": "anthropic/primary",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "gpta-*": {
                "model": "anthropic/primary",
                "hidden": False,
            },
        },
    )

    result = _can_object_call_model(
        model="gpta-1",
        llm_router=llm_router,
        models=["anthropic/primary"],
        team_model_aliases=None,
        object_type="key",
        fallback_depth=0,
    )

    assert result is True




@pytest.mark.parametrize("scope", ["key", "team", "project"])
def test_wildcard_target_only_rpm_tpm_descriptors(scope):
    aliases = {"client-*": "target-*"}
    limits = {"model_rpm_limit": {"target-a": 3}, "model_tpm_limit": {"target-a": 17}}
    auth = UserAPIKeyAuth(**{
        "api_key": "synthetic-key", "team_id": "synthetic-team", "project_id": "synthetic-project",
        "metadata" if scope == "key" else scope + "_metadata": limits,
    })
    handler = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(DualCache()),
        model_group_resolver=lambda model: resolve_model_group_alias(aliases, model),
    )
    descriptors = []
    if scope == "key":
        handler._add_model_per_key_rate_limit_descriptor(auth, "client-a", descriptors)
    elif scope == "team":
        handler._add_team_model_rate_limit_descriptor_from_metadata(auth, "client-a", descriptors)
    else:
        handler._add_project_model_rate_limit_descriptor_from_metadata(auth, "client-a", descriptors)
    assert len(descriptors) == 1
    assert descriptors[0]["key"] == "model_per_" + scope
    assert descriptors[0]["value"].endswith(":target-a")
    assert descriptors[0]["rate_limit"]["requests_per_unit"] == 3
    assert descriptors[0]["rate_limit"]["tokens_per_unit"] == 17


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_hierarchical_wildcard_rewrite_authorizes_target(monkeypatch, allowed):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.user_api_key_auth import _resolve_router_settings_model_group_alias

    router = Router(model_list=[{
        "model_name": "target-a", "litellm_params": {"model": "openai/fork-unmapped", "api_key": "synthetic-key"},
    }])
    monkeypatch.setattr(proxy_server, "llm_router", router)
    token = UserAPIKeyAuth(
        models=["target-a" if allowed else "client-a"],
        router_settings={"model_group_alias": {"client-*": "target-*"}},
    )
    data = {"model": "client-a"}
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    request._body = json.dumps(data).encode()
    request.scope["parsed_body"] = (("model",), dict(data))
    await _resolve_router_settings_model_group_alias(data, token, request, "/v1/chat/completions")
    assert data["model"] == "target-a"
    if allowed:
        assert _can_object_call_model(
            model=data["model"], llm_router=router, models=token.models, team_model_aliases=None,
            object_type="key", fallback_depth=0,
        ) is True
    else:
        from litellm.proxy._types import ModelAccessDeniedProxyException

        with pytest.raises(ModelAccessDeniedProxyException):
            _can_object_call_model(
                model=data["model"], llm_router=router, models=token.models, team_model_aliases=None,
                object_type="key", fallback_depth=0,
            )
