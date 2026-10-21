import pytest

from litellm import Router
from litellm.router_utils.common_utils import resolve_model_group_alias


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


@pytest.mark.parametrize(
    "alias_map, expected",
    [
        ({"group-*": "target-*"}, "target-a"),
        ({"group-*": {"model": "target-*", "hidden": True}}, "target-a"),
        ({"group-*": "target-*", "group-a": "exact"}, "exact"),
        ({"*": "broad", "group-*": "specific"}, "specific"),
        ({"group-*": {"model": 5}, "*": "fallback"}, "fallback"),
    ],
)
def test_wildcard_alias_target_matches_router_for_shared_consumers(alias_map, expected):
    router = Router(model_list=[])
    router.model_group_alias = alias_map
    assert resolve_model_group_alias(alias_map, "group-a") == expected
    assert router._get_model_from_alias("group-a") == expected


@pytest.mark.parametrize("price", [None, 0.0, 0.0002])
@pytest.mark.parametrize("hidden", [False, True])
def test_wildcard_alias_listing_and_group_info_preserve_target_pricing(price, hidden):
    router = Router(
        model_list=[{
            "model_name": "target-a",
            "litellm_params": {"model": "openai/fork-unmapped", "api_key": "test-key"},
            "model_info": {"input_cost_per_token": price, "output_cost_per_token": price},
        }],
        model_group_alias={"group-*": {"model": "target-*", "hidden": hidden}},
    )
    target_rows = router.get_model_list(model_name="target-a")
    alias_rows = router.get_model_list(model_name="group-a")
    assert alias_rows
    assert [row["model_info"] for row in alias_rows] == [row["model_info"] for row in target_rows]
    assert all(row["model_name"] == "group-a" for row in alias_rows)
    target_info = router.get_model_group_info("target-a")
    alias_info = router.get_model_group_info("group-a")
    internal_alias_info = router.get_model_group_info("group-a", include_hidden=True)
    assert internal_alias_info is not None
    assert internal_alias_info.input_cost_per_token == target_info.input_cost_per_token
    assert internal_alias_info.output_cost_per_token == target_info.output_cost_per_token
    if hidden:
        assert alias_info is None
        assert not any(row["model_name"] == "group-*" for row in router.get_model_list())
    else:
        assert alias_info is not None
        assert alias_info.model_group == "group-a"
        assert alias_info.input_cost_per_token == target_info.input_cost_per_token
        assert alias_info.output_cost_per_token == target_info.output_cost_per_token
        if price is not None:
            assert alias_info.input_cost_per_token == price
            assert alias_info.output_cost_per_token == price


@pytest.mark.parametrize("pricing", ["unpriced", "free", "paid", "ptu"])
def test_wildcard_budget_classification_matches_target(pricing):
    from litellm.proxy.auth.auth_checks import _has_ptu_flat_cost, _is_model_cost_zero, model_has_no_cost_mapping

    model_info = {} if pricing == "unpriced" else {
        "input_cost_per_token": 0.001 if pricing == "paid" else 0.0,
        "output_cost_per_token": 0.001 if pricing == "paid" else 0.0,
    }
    if pricing == "ptu":
        model_info.update({"ptu_count": 2, "cost_per_ptu_per_hour": 3.0})
    router = Router(model_list=[{
        "model_name": "target-a",
        "litellm_params": {"model": "openai/fork-" + pricing, "api_key": "synthetic-key"},
        "model_info": model_info,
    }], model_group_alias={"client-*": "target-*"})
    for model in ("target-a", "client-a"):
        assert _has_ptu_flat_cost(router.get_model_list(model_name=model)) is (pricing == "ptu")
        assert _is_model_cost_zero(model=model, llm_router=router) is (pricing == "free")
        assert model_has_no_cost_mapping(model=model, llm_router=router) is (pricing == "unpriced")
