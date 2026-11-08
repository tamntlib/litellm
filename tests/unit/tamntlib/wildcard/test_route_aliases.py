from typing import Final

import pytest

from litellm import Router
from litellm.proxy.route_llm_request import ProxyModelNotFoundError, route_request


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_model", ["gpta-1", "unmatched-1"])
async def test_route_request_routes_only_matching_wildcard_aliases(requested_model: str) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {
                    "model": "openai/synthetic-target",
                    "api_key": "synthetic-key",
                    "mock_response": "selected wildcard target",
                },
                "model_info": {"id": "synthetic-target-deployment"},
            }
        ],
        model_group_alias={"gpta-*": "anthropic/primary"},
    )
    data: Final = {"model": requested_model, "messages": [{"role": "user", "content": "hello"}]}

    if requested_model == "unmatched-1":
        with pytest.raises(ProxyModelNotFoundError):
            await route_request(data, router, None, "acompletion")
        return

    call: Final = await route_request(data, router, None, "acompletion")
    response: Final = await call
    assert response.choices[0].message.content == "selected wildcard target"
    assert response._hidden_params["model_id"] == "synthetic-target-deployment"
    assert data["model"] == requested_model
