import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, Usage


def test_completion_cost_uses_explicit_base_model_pricing_provider(monkeypatch):
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
    response = ModelResponse(
        id="test-id",
        model="anthropic/primary",
        choices=[],
        usage=Usage(prompt_tokens=307, completion_tokens=5, total_tokens=312),
    )

    cost = completion_cost(
        completion_response=response,
        model="anthropic/primary",
        custom_llm_provider="anthropic",
        base_model=mock_base_model,
    )

    assert cost == pytest.approx(0.317)


@pytest.mark.parametrize(
    ("base_model", "transport_provider", "expected_provider"),
    (
        ("azure/gpt-5.6-sol", "anthropic", "azure"),
        ("gpt-4o-mini", "azure", "azure"),
        ("anthropic.claude-3-5-haiku-20241022-v1:0", "bedrock", "bedrock"),
    ),
)
def test_explicit_base_model_preserves_provider_specific_pricing(
    base_model, transport_provider, expected_provider
):
    from litellm.cost_calculator import (
        _resolve_model_name_and_provider_for_cost_calc,
    )

    cost_model, pricing_provider = _resolve_model_name_and_provider_for_cost_calc(
        selected_model=base_model,
        model=f"{transport_provider}/deployment",
        base_model=base_model,
        custom_pricing=False,
        custom_llm_provider=transport_provider,
    )

    assert cost_model == base_model
    assert pricing_provider == expected_provider


def test_providerless_exact_base_model_entry_falls_back_to_transport_provider(
    monkeypatch,
):
    from litellm.cost_calculator import (
        _resolve_model_name_and_provider_for_cost_calc,
    )

    monkeypatch.setitem(
        litellm.model_cost,
        "custom-base-model",
        {
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
            "litellm_provider": None,
        },
    )
    cost_model, pricing_provider = _resolve_model_name_and_provider_for_cost_calc(
        selected_model="custom-base-model",
        model="anthropic/primary",
        base_model="custom-base-model",
        custom_pricing=False,
        custom_llm_provider="anthropic",
    )

    assert cost_model == "custom-base-model"
    assert pricing_provider == "anthropic"


def test_invalid_cross_provider_base_model_falls_back_to_transport_response_model():
    response = ModelResponse(
        id="test-id",
        model="claude-haiku-4-5-20251001",
        choices=[],
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )

    expected_cost = completion_cost(
        completion_response=response,
        model="anthropic/primary",
        custom_llm_provider="anthropic",
    )
    fallback_cost = completion_cost(
        completion_response=response,
        model="anthropic/primary",
        custom_llm_provider="anthropic",
        base_model="openai/not-real",
    )

    assert fallback_cost == expected_cost
    assert fallback_cost > 0


@pytest.mark.parametrize("transport_provider", ["litellm_proxy", "anthropic"])
@pytest.mark.parametrize("base_model", ["openai/estimate-base", "estimate-base"])
def test_failed_input_estimate_matches_cross_provider_success(
    monkeypatch: pytest.MonkeyPatch, transport_provider: str, base_model: str
) -> None:
    from litellm.proxy.utils import _estimated_input_cost

    monkeypatch.setitem(
        litellm.model_cost,
        "estimate-base",
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    model = f"{transport_provider}/estimate-deployment"
    response = ModelResponse(model=model, choices=[], usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10))
    actual_input_cost = completion_cost(
        completion_response=response, model=model, custom_llm_provider=transport_provider, base_model=base_model
    )
    estimated_input_cost = _estimated_input_cost(
        {"model": model, "custom_llm_provider": transport_provider}, 10, deployment_id=None, base_model=base_model
    )

    assert actual_input_cost == pytest.approx(10 * 0.001)
    assert estimated_input_cost == pytest.approx(actual_input_cost)


@pytest.mark.parametrize(("prompt_tokens", "rate", "tier_rate"), [(10, 0.0, 0.0), (10, 0.003, 0.007), (250_000, 0.003, 0.007)])
def test_failed_input_estimate_keeps_deployment_prices_ahead_of_cross_provider_base(
    monkeypatch: pytest.MonkeyPatch, prompt_tokens: int, rate: float, tier_rate: float
) -> None:
    from litellm.proxy.utils import _estimated_input_cost

    monkeypatch.setitem(litellm.model_cost, "estimate-deployment-id", {
        "litellm_provider": "anthropic", "mode": "chat", "input_cost_per_token": rate,
        "input_cost_per_token_above_200k_tokens": tier_rate, "output_cost_per_token": 0.0,
    })
    monkeypatch.setitem(litellm.model_cost, "estimate-custom-base", {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.5, "output_cost_per_token": 0.0,
    })
    model = "anthropic/estimate-deployment"
    response = ModelResponse(model=model, choices=[], usage=Usage(
        prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens
    ))
    actual_input_cost = completion_cost(
        completion_response=response, model=model, custom_llm_provider="anthropic", custom_pricing=True,
        router_model_id="estimate-deployment-id", base_model="openai/estimate-custom-base",
    )
    estimated_input_cost = _estimated_input_cost(
        {"model": model, "custom_llm_provider": "anthropic"}, prompt_tokens,
        deployment_id="estimate-deployment-id", base_model="openai/estimate-custom-base",
    )

    assert actual_input_cost == pytest.approx(prompt_tokens * (tier_rate if prompt_tokens > 200_000 else rate))
    assert estimated_input_cost == pytest.approx(actual_input_cost)


@pytest.mark.parametrize("transport_provider", ["anthropic", "litellm_proxy"])
@pytest.mark.parametrize("base_model", [None, "openai/estimate-nonexistent"])
def test_failed_input_estimate_invalid_base_keeps_transport_fallback(
    monkeypatch: pytest.MonkeyPatch, transport_provider: str, base_model: str | None
) -> None:
    from litellm.proxy.utils import _estimated_input_cost

    model = f"{transport_provider}/estimate-fallback"
    monkeypatch.setitem(litellm.model_cost, model, {
        "litellm_provider": transport_provider, "mode": "chat", "input_cost_per_token": 0.004,
        "output_cost_per_token": 0.0,
    })
    response = ModelResponse(model=model, choices=[], usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10))
    actual_input_cost = completion_cost(
        completion_response=response, model=model, custom_llm_provider=transport_provider, base_model=base_model
    )
    estimated_input_cost = _estimated_input_cost(
        {"model": model, "custom_llm_provider": transport_provider}, 10, deployment_id=None, base_model=base_model
    )

    assert actual_input_cost == pytest.approx(10 * 0.004)
    assert estimated_input_cost == pytest.approx(actual_input_cost)


@pytest.mark.parametrize("transport_provider", ["anthropic", "litellm_proxy"])
@pytest.mark.parametrize("base_model", ["openai/failure-lift-base", "failure-lift-base"])
def test_failure_lift_prices_selected_cross_provider_base_like_success(monkeypatch, transport_provider, base_model):
    from datetime import datetime

    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.utils import _failure_fields_to_lift

    monkeypatch.setitem(litellm.model_cost, "failure-lift-base", {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.002,
        "output_cost_per_token": 0.0,
    })
    model = f"{transport_provider}/failure-lift-deployment"
    messages = [{"role": "user", "content": "synthetic failure input"}]
    logging_obj = Logging(
        model=model, messages=messages, stream=False, call_type="acompletion", start_time=datetime(2020, 1, 1),
        litellm_call_id="fork-failure-call", function_id="fork-failure-function",
    )
    logging_obj.update_environment_variables(
        model=model, user="", optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"base_model": base_model}}),
    )
    logging_obj.model_call_details["first_api_call_start_time"] = datetime(2020, 1, 1)
    logging_obj.model_call_details["messages"] = messages
    logging_obj.model_call_details["call_type"] = "acompletion"
    logging_obj.model_call_details["custom_llm_provider"] = transport_provider
    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)
    usage = lifted["combined_usage_object"]
    actual_input_cost = litellm.completion_cost(
        completion_response=litellm.ModelResponse(model=model, choices=[], usage=usage), model=model,
        custom_llm_provider=transport_provider, base_model=base_model,
    )

    assert usage.prompt_tokens > 0
    assert actual_input_cost == pytest.approx(usage.prompt_tokens * 0.002)
    assert lifted["response_cost"] == pytest.approx(actual_input_cost)


def test_retry_estimates_keep_each_attempts_cross_provider_pricing(monkeypatch):
    from litellm.proxy.utils import _billed_earlier_attempts

    for model, provider, rate in (("retry-first-base", "openai", 0.002), ("retry-final-base", "anthropic", 0.005)):
        monkeypatch.setitem(litellm.model_cost, model, {
            "litellm_provider": provider, "mode": "chat", "input_cost_per_token": rate, "output_cost_per_token": 0.0,
        })
    usage = litellm.Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
    records = (
        {"deployment_id": "retry-first-id", "model": "anthropic/retry-first", "base_model": "openai/retry-first-base",
         "exception_type": "Timeout", "attempted_retries": 0},
        {"deployment_id": "retry-final-id", "model": "litellm_proxy/retry-final",
         "base_model": "retry-final-base", "exception_type": "Timeout", "attempted_retries": 0},
        {"deployment_id": "retry-final-id", "model": "litellm_proxy/retry-final",
         "base_model": "retry-final-base", "exception_type": "Timeout", "attempted_retries": 1},
    )
    attempts = _billed_earlier_attempts(
        model_call_details={"model": "litellm_proxy/retry-final", "custom_llm_provider": "litellm_proxy"},
        litellm_params={"metadata": {"attempted_retries": 1, "previous_models": records}}, estimated_usage=usage,
        base_model="retry-final-base", deployment_id="retry-final-id",
    )

    assert [(a.deployment_id, a.attempted_retries, a.response_cost) for a in attempts] == [
        ("retry-first-id", 0, pytest.approx(10 * 0.002)),
        ("retry-final-id", 0, pytest.approx(10 * 0.005)),
    ]
