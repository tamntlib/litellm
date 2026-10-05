

def test_can_object_call_model_access_via_wildcard_alias_underlying_model():
    from litellm import Router
    from litellm.proxy.auth.auth_checks import _can_object_call_model

    llm_router = Router(
        model_list=[
            {
                "model_name": "anthropic/primary",
                "litellm_params": {
                    "model": "anthropic/primary",
                    "api_key": "test-api-key",
                },
            }
        ],
        model_group_alias={
            "gpta-*": {
                "model": "anthropic/primary",
                "hidden": False,
            },
        },
    )

    result = _can_object_call_model(
        model="gpta-1",
        llm_router=llm_router,
        models=["anthropic/primary"],
        team_model_aliases=None,
        object_type="key",
        fallback_depth=0,
    )

    assert result is True
