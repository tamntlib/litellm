import pytest

from litellm.types.utils import ModelResponse


def test_model_cost_information_uses_base_model_pricing_provider(monkeypatch):
    import litellm
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    mock_base_model = "mock-openai-model"
    monkeypatch.setitem(
        litellm.model_cost,
        mock_base_model,
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "litellm_provider": "openai",
        },
    )

    model_cost_information = StandardLoggingPayloadSetup.get_model_cost_information(
        base_model=mock_base_model,
        custom_pricing=False,
        custom_llm_provider="anthropic",
        init_response_obj=ModelResponse(model="anthropic/primary"),
    )

    assert model_cost_information["model_map_key"] == mock_base_model
    assert model_cost_information["model_map_value"] is not None
    assert model_cost_information["model_map_value"]["litellm_provider"] == "openai"


@pytest.mark.parametrize("transport_rate", [None, 0.0], ids=["capability-only", "explicit-zero"])
@pytest.mark.parametrize("provide_model", [False, True], ids=["response-transport", "request-transport"])
def test_base_pricing_identity_agrees_for_tokens_tools_breakdown_info_and_failure_estimate(
    monkeypatch, transport_rate, provide_model
):
    from datetime import datetime

    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging, StandardLoggingPayloadSetup
    from litellm.proxy.utils import _estimated_input_cost
    from litellm.types.utils import Usage

    base_model = "identity-parity-base"
    model = "openai/identity-parity-deployment"
    base_rates = {
        "input_cost_per_token": 0.003,
        "output_cost_per_token": 0.004,
        "output_cost_per_reasoning_token": 0.006,
        "cache_read_input_token_cost": 0.001,
        "cache_creation_input_token_cost": 0.002,
    }
    monkeypatch.setitem(
        litellm.model_cost,
        base_model,
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            **base_rates,
            "search_context_cost_per_query": {"search_context_size_medium": 0.02},
        },
    )
    transport_entry = {
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": True,
    }
    if transport_rate is not None:
        transport_entry.update({field: transport_rate for field in base_rates})
        transport_entry["search_context_cost_per_query"] = {"search_context_size_medium": 0.0}
    monkeypatch.setitem(litellm.model_cost, f"openai/{base_model}", transport_entry)
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_tokens_details={"cached_tokens": 3, "cache_creation_tokens": 2},
        completion_tokens_details={"reasoning_tokens": 1},
        server_tool_use={"web_search_requests": 2},
    )
    response = ModelResponse(model=model, choices=[], usage=usage)
    logging_obj = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=datetime(2020, 1, 1),
        litellm_call_id="identity-parity",
        function_id="identity-parity",
    )
    cost = litellm.completion_cost(
        completion_response=response,
        model=model if provide_model else None,
        base_model=base_model,
        litellm_logging_obj=logging_obj,
    )
    expected_provider = "anthropic" if transport_rate is None else "openai"
    expected_input = 5 * 0.003 + 3 * 0.001 + 2 * 0.002 if transport_rate is None else 0.0
    expected_output = 3 * 0.004 + 0.006 if transport_rate is None else 0.0
    expected_tools = 2 * 0.02 if transport_rate is None else 0.0
    assert cost == pytest.approx(expected_input + expected_output + expected_tools)
    assert logging_obj.cost_breakdown["input_cost"] == pytest.approx(expected_input)
    assert logging_obj.cost_breakdown["output_cost"] == pytest.approx(expected_output)
    assert logging_obj.cost_breakdown["tool_usage_cost"] == pytest.approx(expected_tools)
    assert logging_obj.cost_breakdown.get("cache_read_cost", 0.0) == pytest.approx(
        3 * 0.001 if transport_rate is None else 0.0
    )
    assert logging_obj.cost_breakdown.get("cache_creation_cost", 0.0) == pytest.approx(
        2 * 0.002 if transport_rate is None else 0.0
    )
    assert logging_obj.cost_breakdown.get("reasoning_cost", 0.0) == pytest.approx(
        0.006 if transport_rate is None else 0.0
    )
    assert logging_obj.billed_token_rates.input_cost_per_token == (
        base_rates["input_cost_per_token"] if transport_rate is None else 0.0
    )
    info = StandardLoggingPayloadSetup.get_model_cost_information(
        base_model=base_model,
        custom_pricing=False,
        custom_llm_provider="openai",
        init_response_obj=response,
    )
    assert info["model_map_value"]["litellm_provider"] == expected_provider
    assert info["model_map_value"]["input_cost_per_token"] == (
        base_rates["input_cost_per_token"] if transport_rate is None else 0.0
    )
    estimate = _estimated_input_cost(
        {"model": model},
        10,
        deployment_id=None,
        base_model=base_model,
    )
    assert estimate == pytest.approx(10 * 0.003 if transport_rate is None else 0.0)
