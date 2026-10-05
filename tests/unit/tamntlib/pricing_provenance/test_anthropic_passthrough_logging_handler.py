from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)


class TestAzureAnthropicCostCalculation:
    def _create_mock_logging_obj(
        self, model: str = None, custom_llm_provider: str = None
    ) -> LiteLLMLoggingObj:
        """Create a mock logging object with optional model and custom_llm_provider"""
        mock_logging_obj = MagicMock()
        mock_model_call_details = {}
        if model:
            mock_model_call_details["model"] = model
        if custom_llm_provider:
            mock_model_call_details["custom_llm_provider"] = custom_llm_provider
        mock_logging_obj.model_call_details = mock_model_call_details
        mock_logging_obj.litellm_call_id = "test-call-id"
        return mock_logging_obj


    def _create_cross_provider_base_model_logging_obj(
        self, monkeypatch, base_model="mock-openai-model"
    ):
        import litellm
        from litellm.proxy import proxy_server

        router = litellm.Router(model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {"model": "anthropic/claude-3-haiku-20240307"},
                "model_info": {"id": "primary-id", "base_model": "gpt-4o-mini"},
            },
            {
                "model_name": "anthropic/fallback",
                "litellm_params": {"model": "anthropic/claude-3-haiku-20240307"},
                "model_info": {"id": "fallback-id", "base_model": base_model},
            },
        ])
        monkeypatch.setattr(proxy_server, "llm_router", router)
        litellm_params = {
            "base_model": "gpt-4o-mini",
            "metadata": {"model_info": {"base_model": "gpt-4o-mini"}},
            "litellm_metadata": {
                "model_group": "anthropic/primary",
                "model_info": {"base_model": "gpt-4o-mini"},
            }
        }
        logging_obj = self._create_mock_logging_obj(
            model="anthropic/primary",
            custom_llm_provider="anthropic",
        )
        logging_obj.model_call_details["litellm_params"] = litellm_params
        logging_obj.litellm_params = litellm_params
        logging_obj.get_router_model_id.return_value = "fallback-id"
        return logging_obj


    @pytest.mark.parametrize("missing", ["router", "deployment", "id", "base_model"])
    @pytest.mark.parametrize("spoof", ["root", "metadata", "litellm_metadata"])
    def test_missing_trusted_base_model_ignores_client_pricing_model(self, monkeypatch, missing, spoof):
        import litellm
        from litellm.proxy import proxy_server
        from litellm.types.utils import ModelResponse, Usage

        logging_obj = self._create_cross_provider_base_model_logging_obj(monkeypatch, base_model=None)
        if missing == "router":
            monkeypatch.setattr(proxy_server, "llm_router", None)
        elif missing == "deployment":
            logging_obj.get_router_model_id.return_value = "deleted-deployment"
        elif missing == "id":
            logging_obj.get_router_model_id.return_value = None
        params = (
            {"base_model": "gpt-4o-mini"} if spoof == "root"
            else {spoof: {"model_info": {"base_model": "gpt-4o-mini"}}}
        )
        logging_obj.litellm_params = params
        logging_obj.model_call_details["litellm_params"] = params
        response = ModelResponse(
            model="claude-haiku-4-5-20251001",
            usage=Usage(prompt_tokens=307, completion_tokens=5),
        )
        expected = litellm.completion_cost(completion_response=response, custom_llm_provider="anthropic")
        spoofed = litellm.completion_cost(
            completion_response=response, custom_llm_provider="anthropic", base_model="gpt-4o-mini"
        )
        cost = AnthropicPassthroughLoggingHandler._compute_response_cost(
            response, "anthropic/primary", logging_obj
        )
        assert cost == pytest.approx(expected)
        assert cost != pytest.approx(spoofed)
        assert logging_obj.model_call_details["litellm_params"] == params
        logging_obj.set_cost_breakdown.assert_called_once()


    def test_selected_deployment_custom_pricing_precedes_base_model(self, monkeypatch):
        import litellm
        from litellm.types.utils import ModelResponse, Usage

        logging_obj = self._create_cross_provider_base_model_logging_obj(monkeypatch, base_model="gpt-4o-mini")
        logging_obj.litellm_params["input_cost_per_token"] = 0.01
        logging_obj.litellm_params["output_cost_per_token"] = 0.02
        monkeypatch.setitem(litellm.model_cost, "fallback-id", {
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "litellm_provider": "anthropic",
        })
        response = ModelResponse(
            model="claude-haiku-4-5-20251001",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )
        cost = AnthropicPassthroughLoggingHandler._compute_response_cost(
            response, "anthropic/primary", logging_obj
        )
        assert cost == pytest.approx(0.14)
        breakdown = logging_obj.set_cost_breakdown.call_args.kwargs
        assert breakdown["input_cost"] == pytest.approx(0.1)
        assert breakdown["output_cost"] == pytest.approx(0.04)


    def test_non_streaming_alias_cost_uses_selected_deployment_base_model(
        self, monkeypatch
    ):
        import litellm
        from litellm.types.utils import ModelResponse, Usage

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        mock_base_model = "mock-openai-model"
        monkeypatch.setitem(
            litellm.model_cost,
            mock_base_model,
            {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "litellm_provider": "openai",
                "max_tokens": 4096,
            },
        )
        logging_obj = self._create_cross_provider_base_model_logging_obj(
            monkeypatch, base_model=mock_base_model
        )
        response = ModelResponse(
            id="test-id",
            model="anthropic/primary",
            choices=[],
            usage=Usage(prompt_tokens=307, completion_tokens=5, total_tokens=312),
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=response,
            model="anthropic/primary",
            kwargs={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            logging_obj=logging_obj,
        )

        assert kwargs["response_cost"] == pytest.approx(0.317)
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.317)
        assert kwargs["model"] == "anthropic/primary"
        assert "complete_streaming_response" not in logging_obj.model_call_details
        logging_obj.set_cost_breakdown.assert_called_once()
        breakdown = logging_obj.set_cost_breakdown.call_args.kwargs
        assert breakdown["input_cost"] == pytest.approx(0.307)
        assert breakdown["output_cost"] == pytest.approx(0.01)


    def test_streaming_alias_cost_uses_selected_deployment_base_model(
        self, monkeypatch
    ):
        import litellm
        from litellm.types.utils import ModelResponse, Usage

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        mock_base_model = "mock-openai-model"
        monkeypatch.setitem(
            litellm.model_cost,
            mock_base_model,
            {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "litellm_provider": "openai",
                "max_tokens": 4096,
            },
        )
        logging_obj = self._create_cross_provider_base_model_logging_obj(
            monkeypatch, base_model=mock_base_model
        )
        logging_obj.model_call_details["stream"] = True
        response = ModelResponse(
            id="test-id",
            model="anthropic/primary",
            choices=[],
            usage=Usage(prompt_tokens=307, completion_tokens=5, total_tokens=312),
        )

        with patch.object(
            AnthropicPassthroughLoggingHandler,
            "_build_complete_streaming_response",
            return_value=response,
        ):
            result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
                litellm_logging_obj=logging_obj,
                passthrough_success_handler_obj=None,
                url_route="/v1/messages",
                request_body={"model": "anthropic/primary", "stream": True},
                endpoint_type="messages",
                start_time=datetime.now(),
                all_chunks=[],
                end_time=datetime.now(),
            )

        assert result["kwargs"]["response_cost"] == pytest.approx(0.317)
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.317)
        assert logging_obj.model_call_details["complete_streaming_response"] is response
