import fnmatch
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


@dataclass(frozen=True, slots=True)
class _PolicyError:
    status_code: int
    detail: str


@dataclass(frozen=True, slots=True)
class _ClientMinimum:
    name: str
    minimum_text: str
    minimum: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _ModelRule:
    pattern: str
    clients: tuple[_ClientMinimum, ...]


_INVALID_POLICY: Final = _PolicyError(
    500,
    "Invalid MODEL_CLIENT_RESTRICTIONS; expected model-pattern:client/major.minor.patch,...",
)
_MAPPING: Final = TypeAdapter(Mapping[str, object])


def _parse_version(value: str) -> tuple[int, int, int] | None:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value) is None:
        return None
    major, minor, patch = value.split(".")
    try:
        return int(major), int(minor), int(patch)
    except ValueError:
        return None


def _parse_client(value: str) -> _ClientMinimum | _PolicyError:
    name, separator, version = value.strip().rpartition("/")
    minimum: Final = _parse_version(version)
    if not separator or re.fullmatch(r"[^\s/,;:]+(?: [^\s/,;:]+)*", name) is None or minimum is None:
        return _INVALID_POLICY
    return _ClientMinimum(name, version, minimum)


def _parse_rule(value: str) -> _ModelRule | _PolicyError:
    pattern, separator, client_text = value.rpartition(":")
    if not separator or not pattern.strip():
        return _INVALID_POLICY
    clients: Final = tuple(_parse_client(client) for client in client_text.split(","))
    error: Final = next((client for client in clients if isinstance(client, _PolicyError)), None)
    if error is not None:
        return error
    return _ModelRule(pattern.strip(), tuple(client for client in clients if isinstance(client, _ClientMinimum)))


def _field(value: object, name: str) -> object:
    try:
        return _MAPPING.validate_python(value).get(name)
    except ValidationError:
        return None


def _get_requested_model(data: Mapping[str, object]) -> object:
    body: Final = _field(data.get("proxy_server_request"), "body")
    model: Final = _field(body, "model")
    return model if isinstance(model, str) and model else data.get("model")


def _get_user_agent(data: Mapping[str, object]) -> str:
    headers: Final = _field(data.get("proxy_server_request"), "headers")
    candidates: Final = (
        _field(data.get("metadata"), "user_agent"),
        _field(data.get("litellm_metadata"), "user_agent"),
        _field(headers, "user-agent"),
        _field(headers, "User-Agent"),
    )
    return next((value.strip() for value in candidates if isinstance(value, str) and value.strip()), "")


def _check_version(client: _ClientMinimum, version_token: str) -> _PolicyError | None:
    version: Final = version_token.split("-")[0].split("+")[0]
    current: Final = _parse_version(version)
    if current is None:
        return _PolicyError(400, f"Invalid {client.name} version")
    if current < client.minimum:
        return _PolicyError(
            400,
            f"{client.name} version must be >= {client.minimum_text}. "
            f"Please update {client.name} to the latest version.",
        )
    return None


def _check_client(rule: _ModelRule, user_agent: str) -> _PolicyError | None:
    if not user_agent:
        return _PolicyError(403, "Missing User-Agent header")
    candidates: Final = tuple(
        (client, re.search(rf"(?:^|\s){re.escape(client.name)}/(\S*)", user_agent, re.IGNORECASE))
        for client in rule.clients
    )
    return next(
        (_check_version(client, match.group(1)) for client, match in candidates if match is not None),
        _PolicyError(403, "Only configured clients are allowed for this model"),
    )


def _check_policy(data: Mapping[str, object], policy: str) -> _PolicyError | None:
    model: Final = _get_requested_model(data)
    if not isinstance(model, str) or not model:
        return None
    rules: Final = tuple(_parse_rule(value.strip()) for value in policy.split(";") if value.strip())
    error: Final = next((rule for rule in rules if isinstance(rule, _PolicyError)), None)
    if error is not None:
        return error
    for rule in rules:
        if isinstance(rule, _ModelRule) and fnmatch.fnmatchcase(model, rule.pattern):
            return _check_client(rule, _get_user_agent(data))
    return None


class ModelClientRestrictionHook(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        error: Final = _check_policy(data, os.getenv("MODEL_CLIENT_RESTRICTIONS", ""))
        if error is not None:
            raise HTTPException(status_code=error.status_code, detail=error.detail)
        return data


proxy_handler_instance: Final = ModelClientRestrictionHook()
