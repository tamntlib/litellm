import os

import litellm
from litellm import verbose_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

VISION_ALIAS_SUFFIX = "-vision"
VISION_ROUTER_MODELS_ENV_VAR = "LITELLM_VISION_MODEL_ROUTER_MODELS"


class AliasAwareVisionModelRouter(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        try:
            vision_router_models = _get_configured_vision_router_models()
            if not vision_router_models:
                return None

            request_model = data.get("model")
            model = _resolve_model(request_model)
            if model is None or model not in vision_router_models:
                return None

            if not has_vision(data, call_type):
                if request_model != model:
                    return {**data, "model": model}
                return None

            return {**data, "model": f"{model}{VISION_ALIAS_SUFFIX}"}
        except Exception:
            verbose_logger.exception("AliasAwareVisionModelRouter: failed to route model")
            return None


def has_vision(data: dict, call_type: str) -> bool:
    if call_type == "anthropic_messages":
        return _anthropic_has_vision(data)
    if call_type == "acompletion":
        return _openai_chat_has_vision(data)
    if call_type == "aresponses":
        return _openai_responses_has_vision(data)
    return False


def _get_configured_vision_router_models() -> set[str]:
    models = os.getenv(VISION_ROUTER_MODELS_ENV_VAR, "")
    return {model.strip() for model in models.split(",") if model.strip()}


def _resolve_model(model: object) -> str | None:
    if not isinstance(model, str):
        return None

    if model in litellm.model_alias_map:
        return litellm.model_alias_map[model]

    router_model = _resolve_router_model_alias(model)
    if router_model is not None:
        return router_model

    return model


def _resolve_router_model_alias(model: str) -> str | None:
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:
        return None

    if llm_router is None:
        return None

    try:
        return llm_router._get_model_from_alias(model)
    except Exception:
        return None


def _anthropic_has_vision(data: dict) -> bool:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "image":
                return True

            nested_content = block.get("content")
            if not isinstance(nested_content, list):
                continue

            for nested_block in nested_content:
                if isinstance(nested_block, dict) and nested_block.get("type") == "image":
                    return True

    return False


def _openai_chat_has_vision(data: dict) -> bool:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return True

    return False


def _openai_responses_has_vision(data: dict) -> bool:
    input_items = data.get("input")
    if not isinstance(input_items, list):
        return False

    for item in input_items:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get("type") == "input_image":
                return True

    return False


proxy_handler_instance = AliasAwareVisionModelRouter()
