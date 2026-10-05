import os
from collections.abc import Mapping
from typing import Final, cast

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import get_default_headers
from litellm.types.utils import CallTypes


def _with_user_agent(headers: object, user_agent: str) -> dict[str, object]:
    source: Final[Mapping[str, object]] = cast(Mapping[str, object], headers) if isinstance(headers, Mapping) else {}
    return {
        **{key: value for key, value in source.items() if key.lower() != "user-agent"},
        "User-Agent": user_agent,
    }


def _with_scoped_user_agent(scoped: object, user_agent: str) -> object:
    if isinstance(scoped, Mapping):
        entry: Final = cast(Mapping[str, object], scoped)
        return {**entry, "extra_headers": _with_user_agent(entry.get("extra_headers"), user_agent)}
    if isinstance(scoped, (list, tuple)):
        return tuple(
            _with_scoped_user_agent(entry, user_agent) for entry in cast(list[object] | tuple[object, ...], scoped)
        )
    return scoped


class ClientUserAgentHook(CustomLogger):
    """Forward client UA only when LITELLM_FORWARD_CLIENT_USER_AGENT=true."""

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        if os.environ.get("LITELLM_FORWARD_CLIENT_USER_AGENT", "").lower() != "true":
            return None
        if call_type not in (
            CallTypes.acompletion,
            CallTypes.aresponses,
            CallTypes.responses,
            CallTypes.anthropic_messages,
        ):
            return None
        request: Final = kwargs.get("proxy_server_request")
        if not isinstance(request, Mapping):
            return None
        ingress: Final = cast(Mapping[str, object], request).get("headers")
        client_agent: Final = (
            next(
                (value for key, value in cast(Mapping[str, object], ingress).items() if key.lower() == "user-agent"),
                None,
            )
            if isinstance(ingress, Mapping)
            else None
        )
        default_agent: Final = cast(str, get_default_headers()["User-Agent"])
        safe_client: Final = (
            client_agent.strip()
            if isinstance(client_agent, str) and "\r" not in client_agent and "\n" not in client_agent
            else ""
        )
        user_agent: Final = (
            safe_client
            if safe_client == default_agent or safe_client.endswith(f" {default_agent}")
            else f"{safe_client} {default_agent}"
            if safe_client
            else default_agent
        )
        headers: Final = (
            litellm.headers
            if call_type == CallTypes.acompletion and not kwargs.get("headers") and not kwargs.get("extra_headers")
            else kwargs.get("headers")
        )
        return {
            **kwargs,
            "headers": _with_user_agent(headers, user_agent),
            "extra_headers": _with_user_agent(kwargs.get("extra_headers"), user_agent),
            **(
                {"provider_specific_header": _with_scoped_user_agent(kwargs["provider_specific_header"], user_agent)}
                if "provider_specific_header" in kwargs
                else {}
            ),
        }


proxy_handler_instance: Final = ClientUserAgentHook()
