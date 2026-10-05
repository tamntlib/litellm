from unittest.mock import MagicMock

import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import _CLIENT_PRICING_CONTROL_FIELDS, add_litellm_data_to_request
from litellm.types.utils import CustomPricingLiteLLMParams


def _make_request_mock() -> Request:
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"
    return request_mock


def _user_api_key_auth(metadata=None, team_metadata=None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        metadata=metadata or {},
        team_metadata=team_metadata or {},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/messages", "/anthropic/v1/messages", "/v1/responses", "/v1/threads"]
)
@pytest.mark.parametrize("permission", [None, "key", "team", "body"])
async def test_base_model_ingress_requires_server_permission(path, permission):
    request = _make_request_mock()
    request.url.path = path
    request.url.__str__.return_value = f"http://localhost{path}"
    data = {
        "model": "gpt-4",
        "base_model": "gpt-4o-mini",
        "metadata": {"model_info": {"base_model": "gpt-4o-mini"}},
        "litellm_metadata": {"model_info": {"base_model": "gpt-4o-mini"}},
    }
    if permission == "body":
        data["metadata"]["allow_client_pricing_override"] = True
    updated = await add_litellm_data_to_request(
        data=data,
        request=request,
        user_api_key_dict=_user_api_key_auth(
            metadata={"allow_client_pricing_override": permission == "key"},
            team_metadata={"allow_client_pricing_override": permission == "team"},
        ),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )
    if permission in ("key", "team"):
        assert updated["base_model"] == "gpt-4o-mini"
    else:
        assert "base_model" not in updated
        for bucket in ("metadata", "litellm_metadata"):
            assert "model_info" not in updated.get(bucket, {})


class TestStripClientPricingOverrides:
    def test_pricing_field_set_tracks_pydantic_model(self):
        # The strip set is built from the model so additions are picked up
        # automatically — this test guards against the model and the strip
        # set drifting apart if someone replaces the auto-derivation later.
        assert _CLIENT_PRICING_CONTROL_FIELDS == frozenset({"base_model"}) | frozenset(
            CustomPricingLiteLLMParams.model_fields.keys()
        )
        # Sanity: the obvious top-level pricing fields are in the set.
        for field in (
            "input_cost_per_token",
            "output_cost_per_token",
            "input_cost_per_second",
            "cache_creation_input_token_cost",
        ):
            assert field in _CLIENT_PRICING_CONTROL_FIELDS
