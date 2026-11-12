from typing import Final

import pytest

from litellm import BadRequestError, Router
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias


@pytest.fixture
def model_list():
    return [{"model_name": "anthropic/primary", "litellm_params": {"model": "anthropic/claude-3-haiku-20240307", "api_key": "test-key"}}]


def test_get_model_from_wildcard_alias(model_list):
    router = Router(
        model_list=model_list,
        model_group_alias={"gpta-*": "anthropic/primary"},
    )

    model = router.registry().resolve("gpta-1", kind="alias")

    assert model == "anthropic/primary"


def test_get_model_from_wildcard_alias_substitutes_target_model(model_list):
    router = Router(
        model_list=model_list,
        model_group_alias={"gpta-*": "anthropic/*"},
    )

    model = router.registry().resolve("gpta-primary", kind="alias")

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
    router = Router(model_list=[], model_group_alias=alias_map)
    assert resolve_model_group_alias(alias_map, "group-a") == expected
    assert router.registry().resolve("group-a", kind="alias") == expected


@pytest.mark.parametrize("hidden", [None, False, True])
@pytest.mark.parametrize("suffix", ["-UNEXPECTED", "\n"])
def test_wildcard_alias_resolver_rejects_trailing_suffix(hidden: bool | None, suffix: str) -> None:
    aliases: Final = {"public-*-safe": "target-*" if hidden is None else {"model": "target-*", "hidden": hidden}}

    assert resolve_model_group_alias(aliases, "public-a-safe") == "target-a"
    assert resolve_model_group_alias(aliases, "public-a-safe" + suffix) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", [None, False, True])
@pytest.mark.parametrize("suffix", ["-UNEXPECTED", "\n"])
async def test_router_wildcard_alias_rejects_trailing_suffix(hidden: bool | None, suffix: str) -> None:
    router: Final = Router(
        model_list=[{
            "model_name": "target-a",
            "litellm_params": {"model": "openai/synthetic-target", "api_key": "synthetic-key"},
        }],
        model_group_alias={
            "public-*-safe": "target-*" if hidden is None else {"model": "target-*", "hidden": hidden},
        },
    )
    invalid_model: Final = "public-a-safe" + suffix

    assert (await router.select("public-a-safe"))["model_name"] == "target-a"
    with pytest.raises(BadRequestError):
        await router.select(invalid_model)
    assert router.registry().resolve(invalid_model, kind="alias") is None
    assert router.registry().resolve(invalid_model, kind="group") == invalid_model
    assert router.registry().resolve(invalid_model, kind="recognized") is False
    assert router.registry().deployments(model_name=invalid_model) == []
    assert router.registry().group(invalid_model, include_hidden=True) is None


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
    target_rows = router.registry().deployments(model_name="target-a")
    alias_rows = router.registry().deployments(model_name="group-a")
    assert alias_rows
    assert [row["model_info"] for row in alias_rows] == [row["model_info"] for row in target_rows]
    assert all(row["model_name"] == "group-a" for row in alias_rows)
    target_info = router.registry().group("target-a")
    alias_info = router.registry().group("group-a")
    internal_alias_info = router.registry().group("group-a", include_hidden=True)
    assert internal_alias_info is not None
    assert internal_alias_info.input_cost_per_token == target_info.input_cost_per_token
    assert internal_alias_info.output_cost_per_token == target_info.output_cost_per_token
    if hidden:
        assert alias_info is None
        assert not any(row["model_name"] == "group-*" for row in router.registry().deployments())
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
        assert _has_ptu_flat_cost(router.registry().deployments(model_name=model)) is (pricing == "ptu")
        assert _is_model_cost_zero(model=model, llm_router=router) is (pricing == "free")
        assert model_has_no_cost_mapping(model=model, llm_router=router) is (pricing == "unpriced")


@pytest.mark.parametrize(
    "aliases, requested, expected",
    [
        ({"client-*::*": "target-*/*"}, "client-a::b", "target-a/b"),
        ({"client.[x]-*": "target-*"}, "client.[x]-a", "target-a"),
        ({"client.[x]-*": "target-*"}, "clientZx-a", None),
        ({"client-*": "target-*"}, "client-", "target-"),
        ({"client-*": "target-*", "client-a": "exact"}, "client-a", "exact"),
        ({"client-a": {"model": 5}, "client-*": "target-*"}, "client-a", "target-a"),
        ({"client-*": {"model": ""}, "*": "fallback"}, "client-a", "fallback"),
        ({"client-*": None, "*": "fallback"}, "client-a", "fallback"),
        ({5: "bad-key", "client-*": "target-*"}, "client-a", "target-a"),
        ({"c*ient-*": "first", "client-*": "second"}, "client-a", "first"),
        ({"client-*": "second", "c*ient-*": "first"}, "client-a", "first"),
        ({"client*": "first", "clien*a": "second"}, "client-a", "first"),
        ({"clien*a": "second", "client*": "first"}, "client-a", "second"),
    ],
)
def test_shared_alias_captures_specificity_and_malformed_tolerance(
    aliases: object, requested: str, expected: str | None
) -> None:
    assert resolve_model_group_alias(aliases, requested) == expected


@pytest.mark.parametrize("hidden", [False, True])
def test_exact_alias_visibility_does_not_fall_through_to_wildcard(hidden: bool) -> None:
    router = Router(
        model_list=[{
            "model_name": "target-a",
            "litellm_params": {"model": "openai/fork-exact", "api_key": "synthetic-key"},
        }],
        model_group_alias={
            "client-a": {"model": "target-a", "hidden": hidden},
            "client-*": "missing-*",
        },
    )
    assert router.registry().resolve("client-a", kind="alias") == "target-a"
    assert router.registry().resolve("client-a", kind="recognized") is True
    internal_info: Final = router.registry().group("client-a", include_hidden=True)
    assert internal_info is not None
    assert internal_info.model_group == "client-a"
    assert (router.registry().group("client-a") is None) is hidden


@pytest.mark.asyncio
async def test_wildcard_alias_changeset_routes_on_scripted_python_core() -> None:
    from litellm.router_api import ChangeSet
    from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, python_core

    host: Final = ScriptedHost()
    core: Final = python_core(
        host,
        model_list=[deployment("target-a", "a"), deployment("replacement-a", "b")],
        model_group_alias={"client-*": "target-*"},
    )
    assert core.registry().resolve("client-a", kind="recognized") is True
    await core.acompletion(model="client-a", messages=[{"role": "user", "content": "hi"}])
    core.apply(ChangeSet(assign={"model_group_alias": {"client-*": "replacement-*"}}))
    await core.acompletion(model="client-a", messages=[{"role": "user", "content": "hi"}])
    assert host.deployments_called("acompletion") == ["a", "b"]
    assert core.registry().resolve("client-a", kind="alias") == "replacement-a"
    assert core.registry().resolve("unmatched-a", kind="recognized") is False


@pytest.mark.parametrize("hidden", [False, True])
def test_wildcard_alias_materializes_routing_group_without_member_access_grants(hidden: bool) -> None:
    router = Router(
        model_list=[{
            "model_name": "target-a",
            "litellm_params": {"model": "openai/fork-group", "api_key": "synthetic-key"},
            "model_info": {"id": "member-a", "access_groups": ["members-only"], "ptu_count": 2,
                           "cost_per_ptu_per_hour": 3.0},
        }],
        routing_groups=[{"group_name": "pool-a", "models": ["target-a"], "routing_strategy": "simple-shuffle"}],
        model_group_alias={"client-*": {"model": "pool-*", "hidden": hidden}},
    )
    rows: Final = router.registry().deployments("client-a")
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["model_name"] == "client-a"
    assert rows[0]["model_info"]["id"] == "member-a"
    assert "access_groups" not in rows[0]["model_info"]
    assert rows[0]["model_info"]["ptu_count"] == 2
    assert rows[0]["model_info"]["cost_per_ptu_per_hour"] == 3.0
    assert not any(row["model_name"] == "client-a" for row in router.registry().deployments())


@pytest.mark.asyncio
@pytest.mark.parametrize("team_id", ["a", "b"])
async def test_wildcard_alias_group_info_and_selection_keep_team_scope(team_id: str) -> None:
    router = Router(
        model_list=[{
            "model_name": "private-" + team,
            "litellm_params": {"model": "openai/fork-team-" + team, "api_key": "synthetic-key"},
            "model_info": {"id": "deployment-" + team, "team_id": team, "team_public_model_name": "target-a",
                           "input_cost_per_token": rate, "output_cost_per_token": rate},
        } for team, rate in (("a", 0.0001), ("b", 0.0002))],
        model_group_alias={"client-*": "target-*"},
    )
    target: Final = router.registry().group("target-a", team_id=team_id)
    alias: Final = router.registry().group("client-a", team_id=team_id)
    assert target is not None
    assert alias is not None
    assert alias.input_cost_per_token == target.input_cost_per_token
    assert alias.output_cost_per_token == target.output_cost_per_token
    assert alias.model_group == "client-a"
    selected: Final = await router.select("client-a", {"metadata": {"user_api_key_team_id": team_id}})
    assert selected["model_info"]["id"] == "deployment-" + team_id
