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
