import os
from fnmatch import fnmatchcase
from typing import Final

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

ANTHROPIC_ONLY_MODELS_ENV_VAR: Final = "ANTHROPIC_ONLY_MODELS"
OPENAI_ONLY_MODELS_ENV_VAR: Final = "OPENAI_ONLY_MODELS"
ANTHROPIC_MESSAGES_CALL_TYPE: Final = "anthropic_messages"
OPENAI_CALL_TYPES: Final = frozenset(
    {
        "completion",
        "acompletion",
        "responses",
        "aresponses",
    }
)


class EndpointModelRoutingHook(CustomLogger):
    """Restrict configured model patterns to their API protocol family."""

    @staticmethod
    def _get_configured_models(environment_variable: str) -> tuple[str, ...]:
        configured_models = os.getenv(environment_variable, "")
        return tuple(model.strip() for model in configured_models.split(",") if model.strip())

    @staticmethod
    def _matches_model(model: object, patterns: tuple[str, ...]) -> bool:
        return isinstance(model, str) and any(fnmatchcase(model, pattern) for pattern in patterns)

    @staticmethod
    def _get_display_model(data: dict, resolved_model: object) -> object:
        proxy_server_request = data.get("proxy_server_request")
        if isinstance(proxy_server_request, dict):
            body = proxy_server_request.get("body")
            if isinstance(body, dict):
                requested_model = body.get("model")
                if isinstance(requested_model, str) and requested_model:
                    return requested_model
        return resolved_model

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral | str,
    ) -> dict:
        anthropic_only_models = self._get_configured_models(ANTHROPIC_ONLY_MODELS_ENV_VAR)
        openai_only_models = self._get_configured_models(OPENAI_ONLY_MODELS_ENV_VAR)

        # Empty lists are intentionally unrestricted. This also makes the hook
        # safe to enable by default in installations that do not configure it.
        if not anthropic_only_models and not openai_only_models:
            return data

        model = data.get("model")
        display_model = self._get_display_model(data, model)
        normalized_call_type = getattr(call_type, "value", call_type)

        if self._matches_model(model, anthropic_only_models) and normalized_call_type != ANTHROPIC_MESSAGES_CALL_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{display_model}' is only available through /v1/messages",
            )

        if self._matches_model(model, openai_only_models) and normalized_call_type not in OPENAI_CALL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{display_model}' is only available through OpenAI endpoints",
            )

        return data


proxy_handler_instance = EndpointModelRoutingHook()
