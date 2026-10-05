from unittest.mock import MagicMock

import pytest

from litellm.proxy.route_llm_request import route_request


@pytest.mark.asyncio
async def test_route_request_routes_wildcard_model_group_alias():
    data = {"model": "gpta-1", "messages": [{"role": "user", "content": "hello"}]}
    llm_router = MagicMock()
    llm_router.router_general_settings.pass_through_all_models = False
    llm_router.default_deployment = None
    llm_router.pattern_router.patterns = []
    llm_router.model_names = ["anthropic/primary"]
    llm_router.has_model_id.return_value = False
    llm_router.deployment_names = []
    llm_router.model_group_alias = {"gpta-*": "anthropic/primary"}
    llm_router._get_model_from_alias.return_value = "anthropic/primary"
    llm_router.acompletion.return_value = "fake_response"

    response = await route_request(data, llm_router, None, "acompletion")

    assert response == "fake_response"
    llm_router.acompletion.assert_called_once_with(**data)
