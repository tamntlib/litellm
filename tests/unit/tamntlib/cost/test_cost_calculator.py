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
