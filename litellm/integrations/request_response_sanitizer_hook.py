import argparse
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

from litellm import verbose_logger
from litellm.constants import (
    MAX_STRING_LENGTH_PROMPT_IN_DB as DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB,
)
from litellm.integrations.custom_logger import CustomLogger

_TRUNCATED_PREVIEW_SUFFIX_RE = re.compile(r"\.\.\. \[\+\d+ chars\]$")
_SYNTHETIC_USER_MESSAGE_PREFIXES = [
    "Side conversation boundary.",
    "Another language model started to solve this problem",
]


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncated_preview_suffix(skipped_chars: int) -> str:
    return f"... [+{skipped_chars} chars]"


def _max_prefix_length_for_final_limit(text_length: int, final_limit: int) -> int:
    if final_limit == -1:
        return -1
    if final_limit <= 0:
        return 0
    if text_length <= final_limit:
        return final_limit

    prefix_length = final_limit
    while prefix_length > 0:
        next_prefix_length = final_limit - len(
            _truncated_preview_suffix(text_length - prefix_length)
        )
        next_prefix_length = max(0, min(prefix_length, next_prefix_length))
        if next_prefix_length == prefix_length:
            return prefix_length
        prefix_length = next_prefix_length
    return 0


CONTENT_PREVIEW_MAX_LENGTH = _get_int_env("CONTENT_PREVIEW_MAX_LENGTH", -1)
USER_TEXT_PREVIEW_MAX_LENGTH = _get_int_env(
    "USER_TEXT_PREVIEW_MAX_LENGTH",
    -1,
)


class RequestFormat(str, Enum):
    CLAUDE = "request_claude"
    OPENAI = "request_openai"
    UNKNOWN = "unknown"


class ResponseFormat(str, Enum):
    CLAUDE = "response_claude"
    OPENAI = "response_openai"
    UNKNOWN = "unknown"


class ClaudeRequestPayloadSanitizer:
    SIGNATURE_KEY = "signature"
    TOP_LEVEL_TOOLS_KEY = "tools"
    TOOL_SCHEMA_KEYS = frozenset({"input_schema"})
    INPUT_SUMMARY_KEYS = frozenset(
        {
            "command",
            "description",
            "file_path",
            "limit",
            "offset",
            "pages",
            "prompt",
            "subagent_type",
            "timeout",
        }
    )

    def sanitize(self, payload: dict) -> dict:
        if _is_request_response_sanitizer_disabled():
            return payload
        if not isinstance(payload, dict):
            return {}

        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key == self.TOP_LEVEL_TOOLS_KEY:
                sanitized[key] = self._sanitize_tools(value)
                continue
            if key == "system":
                sanitized[key] = self._sanitize_system(value)
                continue
            if key == "messages":
                sanitized[key] = self._sanitize_messages(value)
                continue
            if key == "proxy_server_request":
                sanitized[key] = _sanitize_proxy_server_request(value)
                continue
            sanitized[key] = value
        return sanitized

    def _sanitize_tools(self, value: Any) -> Any:
        if not isinstance(value, list):
            return value

        changed = False
        sanitized_tools: list[Any] = []
        for tool in value:
            if not isinstance(tool, dict):
                sanitized_tools.append(tool)
                continue

            needs_sanitize = False
            for key in tool:
                if key in self.TOOL_SCHEMA_KEYS:
                    needs_sanitize = True
                    break
                if key == "description" and isinstance(tool[key], str):
                    needs_sanitize = True
                    break

            if not needs_sanitize:
                sanitized_tools.append(tool)
                continue

            changed = True
            sanitized_tool: dict[str, Any] = {}
            for key, nested_value in tool.items():
                if key in self.TOOL_SCHEMA_KEYS:
                    continue
                if key == "description" and isinstance(nested_value, str):
                    sanitized_tool[key] = truncate(nested_value)
                    continue
                sanitized_tool[key] = nested_value
            sanitized_tools.append(sanitized_tool)
        return sanitized_tools if changed else value

    def _sanitize_system(self, value: Any) -> Any:
        if isinstance(value, str):
            return truncate(value)
        if not isinstance(value, list):
            return value

        changed = False
        sanitized_items: list[Any] = []
        for item in value:
            if not isinstance(item, dict):
                sanitized_items.append(item)
                continue

            needs_sanitize = "text" in item and item.get("type") == "text" and isinstance(item.get("text"), str)
            if not needs_sanitize:
                sanitized_items.append(item)
                continue

            changed = True
            sanitized_item: dict[str, Any] = {}
            for key, nested_value in item.items():
                if key == "text" and item.get("type") == "text" and isinstance(nested_value, str):
                    sanitized_item[key] = truncate(nested_value)
                    continue
                sanitized_item[key] = nested_value
            sanitized_items.append(sanitized_item)
        return sanitized_items if changed else value

    def _sanitize_messages(self, value: Any) -> Any:
        if not isinstance(value, list):
            return value

        list_changed = False
        sanitized_messages: list[Any] = []
        for message_index, message in enumerate(value):
            if not isinstance(message, dict):
                sanitized_messages.append(message)
                continue

            content_items = message.get("content")
            content_list: list[Any] | None = (
                content_items if isinstance(content_items, list) and content_items else None
            )
            msg_changed = False
            sanitized_content: Any = content_items
            if content_list is not None:
                sanitized_content = self._sanitize_content_list(value, message_index, content_list)
                if sanitized_content is not content_list:
                    msg_changed = True
            elif isinstance(content_items, str):
                sanitized_content = truncate(content_items)
                if sanitized_content != content_items:
                    msg_changed = True
            else:
                sanitized_messages.append(message)
                continue

            if not msg_changed:
                sanitized_messages.append(message)
                continue

            list_changed = True
            sanitized_message: dict[str, Any] = {}
            for key, nested_value in message.items():
                if key == "content":
                    sanitized_message[key] = sanitized_content
                    continue
                sanitized_message[key] = nested_value
            sanitized_messages.append(sanitized_message)
        return sanitized_messages if list_changed else value

    def _sanitize_content_list(self, messages: list[Any], message_index: int, value: list[Any]) -> list[Any]:
        changed = False
        sanitized_items: list[Any] = []
        for content_index, content in enumerate(value):
            sanitized = self._sanitize_content_block(
                content,
                is_user_written=is_user_written_text(messages, message_index, content_index),
            )
            if sanitized is not content:
                changed = True
            sanitized_items.append(sanitized)
        return sanitized_items if changed else value

    def _sanitize_content_block(self, value: Any, *, is_user_written: bool) -> Any:
        if not isinstance(value, dict):
            return value

        block_type = value.get("type")

        if (
            self.SIGNATURE_KEY not in value
            and not (block_type == "tool_use" and "input" in value)
            and not (block_type == "tool_result" and "content" in value)
            and not (block_type == "thinking" and "thinking" in value)
            and not (block_type == "text" and "text" in value and isinstance(value.get("text"), str))
            and not (block_type == "image" and "source" in value)
        ):
            return value

        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == self.SIGNATURE_KEY:
                continue
            if key == "input" and block_type == "tool_use":
                sanitized[key] = self._prune_tool_use_input(nested_value)
                continue
            if key == "content" and block_type == "tool_result":
                sanitized[key] = self._truncate_tool_result_content(nested_value)
                continue
            if key == "thinking" and block_type == "thinking":
                sanitized[key] = self._truncate_thinking_value(nested_value)
                continue
            if key == "text" and block_type == "text" and isinstance(nested_value, str):
                text_limit = _get_user_text_preview_max_length(nested_value) if is_user_written else None
                sanitized[key] = truncate(nested_value, max_length=text_limit)
                continue
            if key == "source" and block_type == "image":
                sanitized[key] = truncate_strings_in_value(nested_value)
                continue
            sanitized[key] = nested_value
        return sanitized

    def _prune_tool_use_input(self, value: Any) -> Any:
        if isinstance(value, dict):
            summary: dict[str, Any] = {}
            for key, nested_value in value.items():
                if key in self.INPUT_SUMMARY_KEYS:
                    summary[key] = truncate_strings_in_value(nested_value)
                    continue
            return summary
        return json_preview(value)

    def _truncate_tool_result_content(self, value: Any) -> Any:
        if isinstance(value, str):
            return truncate(value)
        if isinstance(value, list):
            return truncate_strings_in_value(value)
        if value is None:
            return None
        if isinstance(value, dict):
            return truncate_strings_in_value(value)
        return json_preview(value)

    def _truncate_thinking_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return truncate(value)
        if isinstance(value, dict):
            return truncate_strings_in_value(value)
        return json_preview(value)


class OpenAIRequestPayloadSanitizer:
    TOOL_SCHEMA_KEYS = frozenset({"parameters"})
    TEXT_KEYS = frozenset({"text", "arguments", "output"})
    CONTENT_KEYS = frozenset({"content", "summary"})
    METADATA_KEYS = frozenset({"client_metadata", "metadata"})
    ENCRYPTED_CONTENT_KEY = "encrypted_content"

    def sanitize(self, payload: dict) -> dict:
        if _is_request_response_sanitizer_disabled():
            return payload
        if not isinstance(payload, dict):
            return {}

        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "litellm_metadata":
                sanitized[key] = value
                continue
            if key == "instructions":
                sanitized[key] = truncate(value) if isinstance(value, str) else value
                continue
            if key == "input" and isinstance(value, str):
                max_length = _get_active_user_text_preview_max_length(value)
                sanitized[key] = truncate(value, max_length=max_length)
                continue
            if key in {"tools", "functions"}:
                sanitized[key] = self._sanitize_tools(value)
                continue
            if key in {"input", "messages"}:
                sanitized[key] = self._sanitize_message_list(value)
                continue
            if key == "proxy_server_request":
                sanitized[key] = _sanitize_proxy_server_request(value)
                continue
            if key in self.METADATA_KEYS:
                sanitized[key] = truncate_strings_in_value(value)
                continue
            sanitized[key] = truncate_strings_in_value(value)
        return sanitized

    def _sanitize_tools(self, value: Any) -> Any:
        if not isinstance(value, list):
            return value

        changed = False
        sanitized_tools: list[Any] = []
        for tool in value:
            sanitized_tool = self._sanitize_tool(tool)
            if sanitized_tool is not tool:
                changed = True
            sanitized_tools.append(sanitized_tool)
        return sanitized_tools if changed else value

    def _sanitize_tool(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key in self.TOOL_SCHEMA_KEYS:
                changed = True
                continue
            if key == "description" and isinstance(nested_value, str):
                truncated = truncate(nested_value)
                sanitized[key] = truncated
                changed = changed or truncated != nested_value
                continue
            if key == "function" and isinstance(nested_value, dict):
                sanitized_function = self._sanitize_tool(nested_value)
                sanitized[key] = sanitized_function
                changed = changed or sanitized_function is not nested_value
                continue
            if key == "tools" and isinstance(nested_value, list):
                sanitized_tools = self._sanitize_tools(nested_value)
                sanitized[key] = sanitized_tools
                changed = changed or sanitized_tools is not nested_value
                continue
            if key == "format" and isinstance(nested_value, dict):
                sanitized_format = self._sanitize_tool_format(nested_value)
                sanitized[key] = sanitized_format
                changed = changed or sanitized_format is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_tool_format(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "definition":
                changed = True
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_message_list(self, value: Any) -> Any:
        if isinstance(value, str):
            return truncate(value)
        if not isinstance(value, list):
            return value

        changed = False
        sanitized_messages: list[Any] = []
        for message_index, message in enumerate(value):
            sanitized_message = self._sanitize_message(message, value, message_index)
            if sanitized_message is not message:
                changed = True
            sanitized_messages.append(sanitized_message)
        return sanitized_messages if changed else value

    def _sanitize_message(self, value: Any, messages: list[Any], message_index: int) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == self.ENCRYPTED_CONTENT_KEY:
                changed = True
                continue
            if key in self.CONTENT_KEYS:
                sanitized_value = self._sanitize_content_value(nested_value, messages, message_index)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value is not nested_value
                continue
            if key == "tools" and isinstance(nested_value, list):
                sanitized_value = self._sanitize_tools(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value is not nested_value
                continue
            if key == "input":
                sanitized_value = truncate_strings_in_value(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            if key == "output" and isinstance(nested_value, list):
                sanitized_value = truncate_strings_in_value(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            if key in self.METADATA_KEYS:
                sanitized_value = truncate_strings_in_value(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value is not nested_value
                continue
            if key == "tool_calls" and isinstance(nested_value, list):
                sanitized_value = self._sanitize_tool_calls(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value is not nested_value
                continue
            if key in self.TEXT_KEYS and isinstance(nested_value, str):
                sanitized_value = truncate(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_content_value(self, value: Any, messages: list[Any], message_index: int) -> Any:
        if isinstance(value, str):
            max_length = (
                _get_active_user_text_preview_max_length(value)
                if _is_openai_user_written_string_content(messages, message_index, value)
                else None
            )
            return truncate(value, max_length=max_length)
        if isinstance(value, list):
            return self._sanitize_content_list(value, messages, message_index)
        if isinstance(value, dict):
            return truncate_strings_in_value(value)
        if value is None:
            return None
        return json_preview(value)

    def _sanitize_content_list(self, value: list[Any], messages: list[Any], message_index: int) -> list[Any]:
        changed = False
        sanitized_items: list[Any] = []
        for content_index, content in enumerate(value):
            sanitized = self._sanitize_content_block(
                content,
                is_user_written=_is_openai_user_written_text(messages, message_index, content_index),
            )
            if sanitized is not content:
                changed = True
            sanitized_items.append(sanitized)
        return sanitized_items if changed else value

    def _sanitize_content_block(self, value: Any, *, is_user_written: bool) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key in self.TEXT_KEYS and isinstance(nested_value, str):
                max_length = _get_user_text_preview_max_length(nested_value) if is_user_written else None
                sanitized_value = truncate(nested_value, max_length=max_length)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            if isinstance(nested_value, str):
                sanitized_value = truncate(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            if isinstance(nested_value, (dict, list)):
                sanitized_value = truncate_strings_in_value(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_tool_calls(self, value: list[Any]) -> list[Any]:
        changed = False
        sanitized_calls: list[Any] = []
        for tool_call in value:
            sanitized = self._sanitize_tool_call(tool_call)
            if sanitized is not tool_call:
                changed = True
            sanitized_calls.append(sanitized)
        return sanitized_calls if changed else value

    def _sanitize_tool_call(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "function" and isinstance(nested_value, dict):
                sanitized_function = self._sanitize_function_call(nested_value)
                sanitized[key] = sanitized_function
                changed = changed or sanitized_function is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_function_call(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "arguments" and isinstance(nested_value, str):
                sanitized_value = truncate(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value


class ClaudeResponsePayloadSanitizer:
    def sanitize(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return self._sanitize_dict(payload)
        return payload

    def _sanitize_dict(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "choices" and isinstance(nested_value, list):
                sanitized_choices = self._sanitize_choices(nested_value)
                sanitized[key] = sanitized_choices
                changed = changed or sanitized_choices is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_choices(self, value: list[Any]) -> list[Any]:
        changed = False
        sanitized_choices: list[Any] = []
        for choice in value:
            sanitized_choice = self._sanitize_choice(choice)
            if sanitized_choice is not choice:
                changed = True
            sanitized_choices.append(sanitized_choice)
        return sanitized_choices if changed else value

    def _sanitize_choice(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "message" and isinstance(nested_value, dict):
                sanitized_message = self._sanitize_message(nested_value)
                sanitized[key] = sanitized_message
                changed = changed or sanitized_message is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_message(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "provider_specific_fields" and isinstance(nested_value, dict):
                sanitized_fields = self._sanitize_provider_specific_fields(nested_value)
                sanitized[key] = sanitized_fields
                changed = changed or sanitized_fields is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_provider_specific_fields(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "thinking_blocks" and isinstance(nested_value, list):
                sanitized_blocks = self._sanitize_thinking_blocks(nested_value)
                sanitized[key] = sanitized_blocks
                changed = changed or sanitized_blocks is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_thinking_blocks(self, value: list[Any]) -> list[Any]:
        changed = False
        sanitized_blocks: list[Any] = []
        for block in value:
            sanitized_block = self._sanitize_thinking_block(block)
            if sanitized_block is not block:
                changed = True
            sanitized_blocks.append(sanitized_block)
        return sanitized_blocks if changed else value

    def _sanitize_thinking_block(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "signature":
                changed = True
                continue
            if key == "thinking" and isinstance(nested_value, str):
                sanitized_value = truncate(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value


class OpenAIResponsePayloadSanitizer:
    def __init__(self, request_sanitizer: Optional[OpenAIRequestPayloadSanitizer] = None):
        self.request_sanitizer = request_sanitizer or OpenAIRequestPayloadSanitizer()

    def sanitize(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return self._sanitize_dict(payload)
        return payload

    def _sanitize_dict(self, value: dict[str, Any]) -> dict[str, Any]:
        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "instructions" and isinstance(nested_value, str):
                sanitized_value = truncate(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value != nested_value
                continue
            if key == "tools" and isinstance(nested_value, list):
                sanitized_tools = self.request_sanitizer._sanitize_tools(nested_value)
                sanitized[key] = sanitized_tools
                changed = changed or sanitized_tools is not nested_value
                continue
            if key == "output" and isinstance(nested_value, list):
                sanitized_output = self._sanitize_output(nested_value)
                sanitized[key] = sanitized_output
                changed = changed or sanitized_output is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value

    def _sanitize_output(self, value: list[Any]) -> list[Any]:
        changed = False
        sanitized_items: list[Any] = []
        for item in value:
            sanitized_item = self._sanitize_output_item(item)
            if sanitized_item is not item:
                changed = True
            sanitized_items.append(sanitized_item)
        return sanitized_items if changed else value

    def _sanitize_output_item(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        changed = False
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "encrypted_content":
                changed = True
                continue
            if key == "summary":
                sanitized_value = truncate_strings_in_value(nested_value)
                sanitized[key] = sanitized_value
                changed = changed or sanitized_value is not nested_value
                continue
            sanitized[key] = nested_value
        return sanitized if changed else value


class ResponsePayloadSanitizer:
    def __init__(
        self,
        claude_sanitizer: Optional[ClaudeResponsePayloadSanitizer] = None,
        openai_sanitizer: Optional[OpenAIResponsePayloadSanitizer] = None,
    ):
        self.claude_sanitizer = claude_sanitizer or ClaudeResponsePayloadSanitizer()
        self.openai_sanitizer = openai_sanitizer or OpenAIResponsePayloadSanitizer()

    def sanitize(self, payload: Any, call_type: Any = None) -> Any:
        if _is_request_response_sanitizer_disabled():
            return payload
        if not isinstance(payload, dict):
            return payload

        response_format = detect_response_format(payload, call_type)
        if response_format == ResponseFormat.CLAUDE:
            return self.claude_sanitizer.sanitize(payload)
        if response_format == ResponseFormat.OPENAI:
            return self.openai_sanitizer.sanitize(payload)
        return payload


def _is_claude_response_payload(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        provider_fields = message.get("provider_specific_fields")
        if isinstance(provider_fields, dict) and isinstance(provider_fields.get("thinking_blocks"), list):
            return True
    return False


def _is_openai_response_payload(payload: dict[str, Any]) -> bool:
    return payload.get("object") == "response" or "output" in payload


_ANTHROPIC_CALL_TYPES = frozenset({"anthropic_messages"})
_OPENAI_CALL_TYPES = frozenset(
    {
        "completion",
        "acompletion",
        "responses",
        "aresponses",
        "text_completion",
        "atext_completion",
    }
)
_OPENAI_ROUTE_MARKERS = (
    "/v1/responses",
    "/responses",
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
)
_ANTHROPIC_ROUTE_MARKERS = ("/anthropic/v1/messages", "/v1/messages")


def sanitize_claude_request_payload(payload: dict) -> dict:
    if _is_request_response_sanitizer_disabled():
        return payload
    return ClaudeRequestPayloadSanitizer().sanitize(payload)


def sanitize_openai_request_payload(payload: dict) -> dict:
    if _is_request_response_sanitizer_disabled():
        return payload
    return OpenAIRequestPayloadSanitizer().sanitize(payload)


def sanitize_response_payload(payload: Any, call_type: Any = None) -> Any:
    if _is_request_response_sanitizer_disabled():
        return payload
    return ResponsePayloadSanitizer().sanitize(payload, call_type)


def detect_request_format(data: dict[str, Any], call_type: Any) -> RequestFormat:
    if call_type in _ANTHROPIC_CALL_TYPES:
        return RequestFormat.CLAUDE
    if call_type in _OPENAI_CALL_TYPES:
        return RequestFormat.OPENAI

    url = _get_proxy_request_url(data)
    if any(marker in url for marker in _ANTHROPIC_ROUTE_MARKERS):
        return RequestFormat.CLAUDE
    if any(marker in url for marker in _OPENAI_ROUTE_MARKERS):
        return RequestFormat.OPENAI

    model = data.get("model")
    if isinstance(model, str):
        normalized_model = model.lower()
        if normalized_model.startswith("anthropic/") or "claude" in normalized_model:
            return RequestFormat.CLAUDE
        if (
            normalized_model.startswith("openai/")
            or normalized_model.startswith("gpt-")
            or normalized_model.startswith("o")
        ):
            return RequestFormat.OPENAI

    return RequestFormat.UNKNOWN


def get_sanitized_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sanitized{input_path.suffix}")


def is_truncated_preview(text: str, max_length: int) -> bool:
    suffix_match = _TRUNCATED_PREVIEW_SUFFIX_RE.search(text)
    if suffix_match is None:
        return False
    return suffix_match.start() == max_length


def _get_content_preview_max_length() -> int:
    return _get_int_env("CONTENT_PREVIEW_MAX_LENGTH", CONTENT_PREVIEW_MAX_LENGTH)


def _get_max_string_length_prompt_in_db() -> int:
    return _get_int_env(
        "MAX_STRING_LENGTH_PROMPT_IN_DB", DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB
    )


def _get_effective_preview_max_length(text: str, configured_max_length: int) -> int:
    db_safe_max_length = _max_prefix_length_for_final_limit(
        len(text), _get_max_string_length_prompt_in_db()
    )
    if configured_max_length == -1:
        return db_safe_max_length
    return min(configured_max_length, db_safe_max_length)


def _get_user_text_preview_max_length(text: str) -> int:
    env_value = os.getenv("USER_TEXT_PREVIEW_MAX_LENGTH")
    if env_value is not None:
        configured_max_length = _get_int_env(
            "USER_TEXT_PREVIEW_MAX_LENGTH", USER_TEXT_PREVIEW_MAX_LENGTH
        )
        return _get_effective_preview_max_length(text, configured_max_length)
    return _get_effective_preview_max_length(text, USER_TEXT_PREVIEW_MAX_LENGTH)


def _is_request_response_sanitizer_disabled() -> bool:
    return _get_content_preview_max_length() == -1 and _get_int_env(
        "USER_TEXT_PREVIEW_MAX_LENGTH", USER_TEXT_PREVIEW_MAX_LENGTH
    ) == -1


def truncate(text: str, max_length: Optional[int] = None) -> str:
    if max_length is None:
        max_length = _get_effective_preview_max_length(
            text, _get_content_preview_max_length()
        )
    if max_length == -1:
        return text
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if is_truncated_preview(text, max_length=max_length):
        return text
    return text[:max_length] + _truncated_preview_suffix(len(text) - max_length)


def truncate_strings_in_value(value: Any, max_length: Optional[int] = None) -> Any:
    if isinstance(value, str):
        return truncate(text=value, max_length=max_length)
    if isinstance(value, dict):
        return {k: truncate_strings_in_value(v, max_length=max_length) for k, v in value.items()}
    if isinstance(value, list):
        return [truncate_strings_in_value(item, max_length=max_length) for item in value]
    return value


def json_preview(value: Any, max_length: Optional[int] = None) -> str:
    return truncate(json.dumps(value, ensure_ascii=False), max_length=max_length)


def is_user_written_text(messages: list[Any], message_index: int, content_index: int) -> bool:
    message = messages[message_index]
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if message_index > 0 and isinstance(messages[message_index - 1], dict):
        if _message_has_tool_payload(messages[message_index - 1]):
            return False
    content_items = message.get("content") or []
    if content_index != len(content_items) - 1:
        return False
    content = content_items[content_index]
    if not isinstance(content, dict) or content.get("type") != "text":
        return False
    text = content.get("text")
    if not isinstance(text, str):
        return False
    return _is_active_user_written_text(text)


def _is_openai_user_written_text(messages: list[Any], message_index: int, content_index: int) -> bool:
    message = messages[message_index]
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content_items = message.get("content") or []
    if content_index != len(content_items) - 1:
        return False
    content = content_items[content_index]
    if not isinstance(content, dict) or content.get("type") not in {"input_text", "text"}:
        return False
    text = content.get("text")
    if not isinstance(text, str):
        return False
    return _is_active_user_written_text(text)


def _is_openai_user_written_string_content(
    messages: list[Any], message_index: int, text: str
) -> bool:
    message = messages[message_index]
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    return _is_active_user_written_text(text)


def _is_active_user_written_text(text: str) -> bool:
    return not text.startswith("<") and not any(
        text.startswith(prefix) for prefix in _SYNTHETIC_USER_MESSAGE_PREFIXES
    )


def _get_active_user_text_preview_max_length(text: str) -> Optional[int]:
    if _is_active_user_written_text(text):
        return _get_user_text_preview_max_length(text)
    return None


def _message_has_tool_payload(message: dict) -> bool:
    for content in message.get("content") or []:
        if isinstance(content, dict) and content.get("type") in {
            "tool_use",
            "tool_result",
        }:
            return True
    return False


def _sanitize_proxy_server_request(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}

    result: dict[str, Any] = {}
    for key, nested_value in value.items():
        if key == "body":
            result[key] = {}
            continue
        result[key] = nested_value
    return result


def _get_proxy_request_url(data: dict[str, Any]) -> str:
    proxy_server_request = data.get("proxy_server_request")
    if not isinstance(proxy_server_request, dict):
        return ""
    url = proxy_server_request.get("url")
    return url.lower() if isinstance(url, str) else ""


def detect_response_format(payload: dict[str, Any], call_type: Any = None) -> ResponseFormat:
    if call_type in _ANTHROPIC_CALL_TYPES:
        return ResponseFormat.CLAUDE
    if call_type in {"responses", "aresponses"}:
        return ResponseFormat.OPENAI
    if _is_openai_response_payload(payload):
        return ResponseFormat.OPENAI
    if _is_claude_response_payload(payload):
        return ResponseFormat.CLAUDE
    return ResponseFormat.UNKNOWN


def _detect_file_payload_format(payload: dict[str, Any]) -> RequestFormat | ResponseFormat:
    response_format = detect_response_format(payload)
    if response_format != ResponseFormat.UNKNOWN:
        return response_format
    if "instructions" in payload or "input" in payload:
        return RequestFormat.OPENAI
    if "system" in payload or "messages" in payload:
        return RequestFormat.CLAUDE
    return RequestFormat.UNKNOWN


def _parse_cli_payload_format(value: str) -> RequestFormat | ResponseFormat:
    try:
        return RequestFormat(value)
    except ValueError:
        return ResponseFormat(value)


class RequestResponseSanitizerHook(CustomLogger):
    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict,
        cache,
        data,
        call_type,
    ) -> Optional[dict]:
        try:
            if not isinstance(data, dict):
                return data

            proxy_server_request = data.get("proxy_server_request")
            if not isinstance(proxy_server_request, dict):
                return data

            body = proxy_server_request.get("body")
            if not isinstance(body, dict):
                return data

            request_format = detect_request_format(data, call_type)
            if request_format == RequestFormat.CLAUDE:
                proxy_server_request["body"] = sanitize_claude_request_payload(body)
                return data
            if request_format == RequestFormat.OPENAI:
                proxy_server_request["body"] = sanitize_openai_request_payload(body)
                return data
        except Exception:
            verbose_logger.exception("RequestResponseSanitizerHook: failed to sanitize request")
        return data

    async def async_logging_hook(self, kwargs: dict, result: Any, call_type: str):  # type: ignore[override]
        try:
            return self._sanitize_logging_payload(kwargs, result, call_type)
        except Exception:
            verbose_logger.exception("RequestResponseSanitizerHook: failed to sanitize response")
        return kwargs, result

    def logging_hook(self, kwargs: dict, result: Any, call_type: str):  # type: ignore[override]
        try:
            return self._sanitize_logging_payload(kwargs, result, call_type)
        except Exception:
            verbose_logger.exception("RequestResponseSanitizerHook: failed to sanitize response")
        return kwargs, result

    def _sanitize_logging_payload(self, kwargs: dict, result: Any, call_type: str):
        standard_logging_object = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            response = standard_logging_object.get("response")
            if response is not None:
                standard_logging_object["response"] = sanitize_response_payload(response, call_type)
        return kwargs, result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitize request JSON by removing spend-log-heavy fields.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument(
        "--format",
        choices=(
            RequestFormat.CLAUDE.value,
            RequestFormat.OPENAI.value,
            ResponseFormat.CLAUDE.value,
            ResponseFormat.OPENAI.value,
        ),
        dest="request_format",
    )
    args = parser.parse_args(argv)

    output_path = args.output_path or get_sanitized_output_path(args.input_path)
    payload = json.loads(args.input_path.read_text(encoding="utf-8"))
    payload_format = (
        _parse_cli_payload_format(args.request_format) if args.request_format else _detect_file_payload_format(payload)
    )
    if payload_format == RequestFormat.OPENAI:
        sanitized_payload = sanitize_openai_request_payload(payload)
    elif payload_format == ResponseFormat.OPENAI:
        sanitized_payload = OpenAIResponsePayloadSanitizer().sanitize(payload)
    elif payload_format == ResponseFormat.CLAUDE:
        sanitized_payload = ClaudeResponsePayloadSanitizer().sanitize(payload)
    else:
        sanitized_payload = sanitize_claude_request_payload(payload)
    output_path.write_text(
        json.dumps(sanitized_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


proxy_handler_instance = RequestResponseSanitizerHook()


if __name__ == "__main__":
    raise SystemExit(main())
