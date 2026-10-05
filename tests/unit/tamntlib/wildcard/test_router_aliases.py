import pytest

from litellm import Router


@pytest.fixture
def model_list():
    return [{"model_name": "anthropic/primary", "litellm_params": {"model": "anthropic/claude-3-haiku-20240307", "api_key": "test-key"}}]


def test_get_model_from_wildcard_alias(model_list):
    router = Router(
        model_list=model_list,
        model_group_alias={"gpta-*": "anthropic/primary"},
    )

    model = router._get_model_from_alias(model="gpta-1")

    assert model == "anthropic/primary"


def test_get_model_from_wildcard_alias_substitutes_target_model(model_list):
    router = Router(
        model_list=model_list,
        model_group_alias={"gpta-*": "anthropic/*"},
    )

    model = router._get_model_from_alias(model="gpta-primary")

    assert model == "anthropic/primary"
