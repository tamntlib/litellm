import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB
from litellm.integrations.custom_logger import CustomLogger

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from litellm.integrations.request_response_sanitizer_hook import (
    CONTENT_PREVIEW_MAX_LENGTH,
    ClaudeRequestPayloadSanitizer,
    ClaudeResponsePayloadSanitizer,
    OpenAIRequestPayloadSanitizer,
    OpenAIResponsePayloadSanitizer,
    RequestFormat,
    RequestResponseSanitizerHook,
    ResponseFormat,
    detect_request_format,
    detect_response_format,
    get_sanitized_output_path,
    is_user_written_text,
    sanitize_claude_request_payload,
    sanitize_openai_request_payload,
    sanitize_response_payload,
    truncate,
    truncate_strings_in_value,
    USER_TEXT_PREVIEW_MAX_LENGTH,
)


@pytest.fixture(autouse=True)
def enable_request_response_sanitizer_for_tests(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "100")
    monkeypatch.setenv(
        "USER_TEXT_PREVIEW_MAX_LENGTH", str(MAX_STRING_LENGTH_PROMPT_IN_DB)
    )


def test_truncate_is_idempotent_for_existing_preview_suffix():
    value = f"{'a' * 100}... [+10 chars]"

    assert truncate(value) == value


def test_truncate_strings_in_value_preserves_shape_and_non_strings():
    nested = {"tool_output": "c" * 131}
    value = {
        "text": "b" * 130,
        "nested": ["short", nested],
        "count": 3,
        "enabled": True,
    }

    sanitized = truncate_strings_in_value(value)

    assert sanitized == {
        "text": f"{'b' * 100}... [+30 chars]",
        "nested": ["short", {"tool_output": f"{'c' * 100}... [+31 chars]"}],
        "count": 3,
        "enabled": True,
    }
    assert value["text"] == "b" * 130
    assert nested["tool_output"] == "c" * 131


def test_preview_limits_default_to_disabled():
    assert CONTENT_PREVIEW_MAX_LENGTH == -1
    assert USER_TEXT_PREVIEW_MAX_LENGTH == -1


def test_sanitize_openai_request_payload_skips_by_default_without_preview_env(monkeypatch):
    monkeypatch.delenv("CONTENT_PREVIEW_MAX_LENGTH", raising=False)
    monkeypatch.delenv("USER_TEXT_PREVIEW_MAX_LENGTH", raising=False)
    payload = {"instructions": "i" * 130}

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized is payload


def test_sanitize_openai_request_payload_uses_short_preview_for_side_conversation_boundary(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "80")
    boundary_text = "Side conversation boundary.\n\n" + ("b" * 120)
    user_text = "u" * 120
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": boundary_text}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert (
        sanitized["input"][0]["content"][0]["text"]
        == f"{boundary_text[:20]}... [+129 chars]"
    )
    assert sanitized["input"][1]["content"][0]["text"] == f"{'u' * 80}... [+40 chars]"


def test_sanitize_openai_request_payload_uses_short_preview_for_compaction_summary(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "80")
    summary_text = (
        "Another language model started to solve this problem and produced a summary "
        "of its thinking process.\n"
        + ("s" * 120)
    )
    user_text = "u" * 120
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": summary_text}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert (
        sanitized["input"][0]["content"][0]["text"]
        == f"{summary_text[:20]}... [+{len(summary_text) - 20} chars]"
    )
    assert sanitized["input"][1]["content"][0]["text"] == f"{'u' * 80}... [+40 chars]"


def test_sanitize_openai_request_payload_default_user_preview_fits_db_prompt_limit(monkeypatch):
    monkeypatch.delenv("USER_TEXT_PREVIEW_MAX_LENGTH", raising=False)
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "128")
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "u" * 500}],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)
    sanitized_text = sanitized["input"][0]["content"][0]["text"]

    assert len(sanitized_text) <= 128
    assert sanitized_text.startswith("u")
    assert sanitized_text.endswith(" chars]")


def test_sanitize_openai_request_payload_caps_user_preview_env_to_db_prompt_limit(monkeypatch):
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "1000")
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "128")
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "u" * 500}],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)
    sanitized_text = sanitized["input"][0]["content"][0]["text"]

    assert len(sanitized_text) <= 128
    assert sanitized_text.endswith(" chars]")


def test_sanitize_openai_request_payload_uses_user_preview_for_string_input(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "80")
    payload = {"input": "u" * 120}

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized["input"] == f"{'u' * 80}... [+40 chars]"


def test_sanitize_openai_request_payload_uses_user_preview_for_string_message_content(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "80")
    payload = {
        "messages": [
            {"role": "system", "content": "s" * 120},
            {"role": "user", "content": "u" * 120},
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized["messages"][0]["content"] == f"{'s' * 20}... [+100 chars]"
    assert sanitized["messages"][1]["content"] == f"{'u' * 80}... [+40 chars]"


def test_sanitize_openai_request_payload_caps_content_preview_env_to_db_prompt_limit(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "1000")
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "128")
    payload = {"instructions": "i" * 500}

    sanitized = sanitize_openai_request_payload(payload)
    sanitized_text = sanitized["instructions"]

    assert len(sanitized_text) <= 128
    assert sanitized_text.endswith(" chars]")


def test_sanitize_openai_request_payload_truncates_input_image_url(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "80")
    image_url = "data:image/png;base64," + ("i" * 180)
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "u" * 120},
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": image_url,
                    },
                ],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert (
        sanitized["input"][0]["content"][1]["image_url"]
        == f"{image_url[:20]}... [+{len(image_url) - 20} chars]"
    )


def test_sanitize_openai_request_payload_truncates_nested_content_payloads(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    image_url = "data:image/png;base64," + ("i" * 180)
    audio_data = "a" * 180
    file_data = "data:application/pdf;base64," + ("f" * 180)
    file_url = "https://example.com/" + ("p" * 180)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": "mp3"},
                    },
                    {
                        "type": "file",
                        "file": {"file_data": file_data, "filename": "large.pdf"},
                    },
                    {
                        "type": "input_file",
                        "file_url": file_url,
                    },
                ],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)
    content = sanitized["messages"][0]["content"]

    assert content[0]["image_url"] == {
        "url": f"{image_url[:20]}... [+{len(image_url) - 20} chars]",
        "detail": "high",
    }
    assert content[1]["input_audio"] == {
        "data": f"{audio_data[:20]}... [+{len(audio_data) - 20} chars]",
        "format": "mp3",
    }
    assert content[2]["file"] == {
        "file_data": f"{file_data[:20]}... [+{len(file_data) - 20} chars]",
        "filename": "large.pdf",
    }
    assert (
        content[3]["file_url"]
        == f"{file_url[:20]}... [+{len(file_url) - 20} chars]"
    )


def test_sanitize_openai_request_payload_truncates_output_image_url(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    image_url = "data:image/png;base64," + ("i" * 180)
    payload = {
        "input": [
            {
                "type": "function_call_output",
                "output": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": image_url,
                    }
                ],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert (
        sanitized["input"][0]["output"][0]["image_url"]
        == f"{image_url[:20]}... [+{len(image_url) - 20} chars]"
    )


def test_sanitize_openai_request_payload_truncates_unknown_nested_request_fields():
    payload = {
        "input": "hi",
        "prediction": {"type": "content", "content": "p" * 130},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "description": "d" * 130,
            },
        },
        "litellm_metadata": {"headers": {"x-codex-turn-metadata": "m" * 130}},
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized["prediction"] == {
        "type": "content",
        "content": f"{'p' * 100}... [+30 chars]",
    }
    assert sanitized["response_format"]["json_schema"] == {
        "name": "answer",
        "description": f"{'d' * 100}... [+30 chars]",
    }
    assert sanitized["litellm_metadata"] is payload["litellm_metadata"]


def test_sanitize_openai_request_payload_sanitizes_legacy_functions():
    payload = {
        "functions": [
            {
                "name": "search",
                "description": "d" * 130,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized == {
        "functions": [
            {
                "name": "search",
                "description": f"{'d' * 100}... [+30 chars]",
            }
        ]
    }


def test_sanitize_openai_request_payload_skips_when_preview_limits_are_disabled(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "-1")
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "-1")
    payload = {
        "instructions": "i" * 130,
        "input": [
            {
                "role": "assistant",
                "type": "reasoning",
                "encrypted_content": "secret-reasoning",
                "summary": [{"type": "summary_text", "text": "r" * 130}],
            }
        ],
        "tools": [
            {"type": "function", "name": "search", "parameters": {"type": "object"}}
        ],
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized is payload


def test_get_sanitized_output_path_keeps_suffix_and_adds_marker():
    assert get_sanitized_output_path(Path("request.json")) == Path("request_sanitized.json")


def test_detect_request_format_prefers_anthropic_call_type_over_model_name():
    data = {
        "model": "openai/gpt-5.5",
        "proxy_server_request": {"url": "http://localhost:4000/anthropic/v1/messages"},
    }

    assert detect_request_format(data, "anthropic_messages") == RequestFormat.CLAUDE


def test_detect_request_format_prefers_openai_route_over_claude_model_name():
    data = {
        "model": "claude-sonnet-4-6",
        "proxy_server_request": {"url": "http://localhost:4000/v1/responses"},
    }

    assert detect_request_format(data, "responses") == RequestFormat.OPENAI


def test_detect_request_format_uses_url_when_call_type_is_unknown():
    data = {
        "model": "alias-model",
        "proxy_server_request": {"url": "http://localhost:4000/v1/chat/completions"},
    }

    assert detect_request_format(data, "unknown") == RequestFormat.OPENAI


def test_detect_response_format_returns_provider_specific_response_format():
    claude_payload = {
        "choices": [
            {
                "message": {
                    "provider_specific_fields": {"thinking_blocks": [{"type": "thinking", "signature": "opaque"}]}
                }
            }
        ]
    }
    openai_payload = {
        "object": "response",
        "output": [{"type": "reasoning", "encrypted_content": "opaque"}],
    }

    assert detect_response_format(claude_payload) == ResponseFormat.CLAUDE
    assert detect_response_format(openai_payload) == ResponseFormat.OPENAI
    assert detect_response_format(openai_payload, "anthropic_messages") == ResponseFormat.CLAUDE
    assert detect_response_format(claude_payload, "responses") == ResponseFormat.OPENAI


@pytest.mark.asyncio
async def test_request_sanitizer_hook_dispatches_to_claude_sanitizer():
    hook = RequestResponseSanitizerHook()
    data = {
        "model": "claude-sonnet-4-6",
        "proxy_server_request": {
            "url": "http://localhost:4000/anthropic/v1/messages",
            "body": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "reasoning text",
                                "signature": "opaque-signature",
                            }
                        ],
                    }
                ]
            },
        },
    }

    updated = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert updated is data
    assert data["proxy_server_request"]["body"] == {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "reasoning text"}],
            }
        ]
    }


@pytest.mark.asyncio
async def test_request_sanitizer_hook_dispatches_to_openai_sanitizer_for_openai_route_with_claude_model():
    hook = RequestResponseSanitizerHook()
    data = {
        "model": "claude-sonnet-4-6",
        "proxy_server_request": {
            "url": "http://localhost:4000/v1/responses",
            "body": {
                "instructions": "i" * 130,
                "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            },
        },
    }

    await hook.async_pre_call_hook(None, None, data, "responses")

    assert data["proxy_server_request"]["body"] == {
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [{"type": "function", "name": "search"}],
    }


def test_sanitize_claude_request_payload_removes_known_content_signature_fields():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                        "signature": "opaque-signature",
                    }
                ],
            }
        ],
        "metadata": {"signature": "metadata-signature", "trace_id": "trace-123"},
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                    }
                ],
            }
        ],
        "metadata": {"signature": "metadata-signature", "trace_id": "trace-123"},
    }
    assert payload["messages"][0]["content"][0]["signature"] == "opaque-signature"


def test_sanitize_claude_request_payload_preserves_unknown_top_level_fields():
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "metadata": {"trace_id": "trace-123"},
        "stream": True,
        "extra": {"deep": {"signature": "ignored-signature"}},
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized["model"] == "claude-sonnet-4-6"
    assert sanitized["stream"] is True
    assert sanitized["extra"] == {"deep": {"signature": "ignored-signature"}}


def test_sanitize_claude_request_payload_truncates_top_level_tool_descriptions():
    payload = {
        "tools": [
            {
                "name": "search",
                "description": "Long tool description",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                },
            }
        ],
        "metadata": {"description": "keep this metadata description"},
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "tools": [{"name": "search", "description": "Long tool description"}],
        "metadata": {"description": "keep this metadata description"},
    }


def test_sanitize_claude_request_payload_truncates_long_tool_description():
    payload = {
        "tools": [
            {
                "name": "search",
                "description": "a" * 161,
                "input_schema": {"type": "object"},
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "tools": [
            {
                "name": "search",
                "description": f"{'a' * 100}... [+61 chars]",
            }
        ]
    }


def test_sanitize_claude_request_payload_truncates_system_prompt_text():
    payload = {
        "system": [
            {
                "type": "text",
                "text": "s" * 200,
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "system": [
            {
                "type": "text",
                "text": f"{'s' * 100}... [+100 chars]",
            }
        ]
    }


def test_sanitize_claude_request_payload_truncates_string_system_prompt():
    payload = {"system": "s" * 200}

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {"system": f"{'s' * 100}... [+100 chars]"}


def test_sanitize_claude_request_payload_truncates_string_message_content():
    payload = {
        "messages": [
            {"role": "system", "content": "s" * 200},
            {"role": "assistant", "content": "a" * 200},
            {"role": "user", "content": "u" * 200},
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "messages": [
            {"role": "system", "content": f"{'s' * 100}... [+100 chars]"},
            {"role": "assistant", "content": f"{'a' * 100}... [+100 chars]"},
            {"role": "user", "content": f"{'u' * 100}... [+100 chars]"},
        ]
    }


def test_sanitize_claude_request_payload_truncates_heavy_message_content():
    long_content = "a" * 161
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {
                            "command": "very long command",
                            "old_string": "large old string",
                            "new_string": "large new string",
                            "todos": [{"content": "large todo payload"}],
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": long_content,
                        "is_error": False,
                    },
                    {"type": "thinking", "thinking": "reasoning text"},
                    {"type": "text", "text": "keep visible answer"},
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {
                            "command": "very long command",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": f"{'a' * 100}... [+61 chars]",
                        "is_error": False,
                    },
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                    },
                    {"type": "text", "text": "keep visible answer"},
                ],
            }
        ]
    }


def test_sanitize_claude_request_payload_truncates_long_tool_use_input_prompt():
    long_prompt = "b" * 161
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Agent",
                        "input": {"prompt": long_prompt, "subagent_type": "research"},
                    }
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    input_value = sanitized["messages"][0]["content"][0]["input"]
    assert input_value == {
        "prompt": f"{'b' * 100}... [+61 chars]",
        "subagent_type": "research",
    }


def test_sanitize_claude_request_payload_preserves_list_tool_result_object_type():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "a" * 200,
                            },
                            {"type": "text", "text": "ok"},
                        ],
                        "is_error": False,
                    }
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    content = sanitized["messages"][0]["content"][0]["content"]
    assert isinstance(content, list)
    assert isinstance(content[0], dict)
    assert content[0]["type"] == "text"
    assert content[0]["text"].endswith("]")
    assert content[1] == {"type": "text", "text": "ok"}


def test_is_user_written_text_detects_last_user_text_not_after_tool_message():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>context</system-reminder>",
                    },
                    {"type": "text", "text": "actual user prompt"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "tool injected text"}],
            },
        ]
    }

    assert is_user_written_text(payload["messages"], 0, 1) is True
    assert is_user_written_text(payload["messages"], 0, 0) is False
    assert is_user_written_text(payload["messages"], 2, 0) is False


def test_sanitize_claude_request_payload_uses_longer_preview_for_user_written_text(monkeypatch):
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "4000")
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "5000")
    text = "a" * 4201
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>context</system-reminder>",
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized["messages"][0]["content"][1]["text"] == f"{'a' * 4000}... [+201 chars]"


def test_sanitize_claude_request_payload_truncates_tagged_text_blocks():
    tagged_text = f"<system-reminder>{'a' * 180}</system-reminder>"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": tagged_text,
                    },
                    {
                        "type": "text",
                        "text": "regular text remains visible",
                    },
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{tagged_text[:100]}... [+{len(tagged_text) - 100} chars]",
                    },
                    {
                        "type": "text",
                        "text": "regular text remains visible",
                    },
                ],
            }
        ]
    }


def test_sanitize_claude_request_payload_truncates_image_source_data(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    image_data = "/9j/" + ("i" * 180)
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "describe this image"},
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": f"{image_data[:20]}... [+{len(image_data) - 20} chars]",
                        },
                    },
                    {"type": "text", "text": "describe this image"},
                ],
            }
        ]
    }
    assert payload["messages"][0]["content"][0]["source"]["data"] == image_data


def test_sanitize_claude_request_payload_truncates_image_source_url(monkeypatch):
    monkeypatch.setenv("CONTENT_PREVIEW_MAX_LENGTH", "20")
    image_url = "https://example.com/" + ("p" * 180)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        },
                    }
                ],
            }
        ]
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized["messages"][0]["content"][0]["source"] == {
        "type": "url",
        "url": f"{image_url[:20]}... [+{len(image_url) - 20} chars]",
    }


@pytest.mark.asyncio
async def test_async_pre_call_hook_sanitizes_proxy_server_request_body():
    hook = RequestResponseSanitizerHook()
    data = {
        "proxy_server_request": {
            "body": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Long tool description",
                        "input_schema": {"type": "object"},
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "reasoning text",
                                "signature": "opaque-signature",
                            }
                        ],
                    }
                ],
            }
        }
    }

    updated = await hook.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="anthropic_messages",
    )

    assert updated is data
    assert data["proxy_server_request"]["body"] == {
        "tools": [{"name": "search", "description": "Long tool description"}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_async_logging_hook_inherited_no_op_does_not_change_body():
    hook = RequestResponseSanitizerHook()
    body = {
        "tools": [
            {
                "name": "search",
                "description": "Long tool description",
                "input_schema": {"type": "object"},
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                        "signature": "opaque-signature",
                    }
                ],
            }
        ],
    }
    kwargs = {
        "litellm_params": {
            "proxy_server_request": {
                "body": body,
            }
        }
    }
    result = {"id": "chatcmpl-test"}

    updated_kwargs, updated_result = await hook.async_logging_hook(kwargs=kwargs, result=result, call_type="completion")

    assert updated_kwargs is kwargs
    assert updated_result is result
    assert updated_kwargs["litellm_params"]["proxy_server_request"]["body"] is body


def test_sanitize_claude_request_payload_preserves_proxy_server_request_header_values():
    payload = {
        "proxy_server_request": {
            "url": "http://example.com",
            "method": "POST",
            "headers": {
                "anthropic-beta": "a" * 200,
            },
        }
    }

    sanitized = sanitize_claude_request_payload(payload)

    assert sanitized == {
        "proxy_server_request": {
            "url": "http://example.com",
            "method": "POST",
            "headers": {
                "anthropic-beta": "a" * 200,
            },
        }
    }


@pytest.mark.asyncio
async def test_pre_call_and_logging_hooks_do_not_double_sanitize_body(monkeypatch):
    import litellm.integrations.request_response_sanitizer_hook as sanitizer_module

    call_count = 0

    def spy(payload):
        nonlocal call_count
        call_count += 1
        return payload

    monkeypatch.setattr(sanitizer_module, "sanitize_claude_request_payload", spy)

    hook = sanitizer_module.RequestResponseSanitizerHook()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    data = {"proxy_server_request": {"body": body}}

    await hook.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="anthropic_messages",
    )
    await hook.async_logging_hook(
        kwargs={"litellm_params": data},
        result={},
        call_type="completion",
    )

    assert call_count == 1


def test_sanitize_claude_request_payload_keeps_proxy_server_request_body_empty():
    base = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
    }
    nested = {"body": base}
    base["proxy_server_request"] = nested  # type: ignore[index]
    sanitized = sanitize_claude_request_payload(base)

    serialized = json.dumps(sanitized)
    assert sanitized["proxy_server_request"]["body"] == {}
    assert "truncated recursive proxy_server_request" not in serialized
    assert "truncated deep proxy_server_request" not in serialized


def test_get_sanitized_output_path():
    assert get_sanitized_output_path(Path("claude-request.json")) == Path("claude-request_sanitized.json")


def test_cli_writes_sanitized_json(tmp_path: Path):
    from litellm.integrations.request_response_sanitizer_hook import main

    input_path = tmp_path / "claude-request.json"
    output_path = tmp_path / "claude-request_sanitized.json"
    input_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "search",
                        "description": "Long tool description",
                        "input_schema": {"type": "object"},
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "reasoning text",
                                "signature": "opaque",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main([str(input_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "tools": [{"name": "search", "description": "Long tool description"}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning text",
                    }
                ],
            }
        ],
    }


def test_sanitize_does_not_mutate_provider_request_payload():
    """Sanitize only affects spend-log snapshot; provider-facing payload stays intact."""
    tool_obj = {
        "name": "search",
        "description": "a" * 200,
        "input_schema": {"type": "object"},
    }
    thinking_block = {
        "type": "thinking",
        "thinking": "reasoning text",
        "signature": "opaque-signature",
    }
    user_text = "b" * 400
    user_text_block = {"type": "text", "text": user_text}
    message_with_thinking = {"role": "assistant", "content": [thinking_block]}
    message_with_user_text = {"role": "user", "content": [user_text_block]}
    metadata = {"signature": "metadata-sig", "trace_id": "trace-123"}
    proxy_server_request = {
        "url": "http://example.com",
        "method": "POST",
        "body": {"model": "claude-sonnet-4-6"},
    }

    payload = {
        "tools": [tool_obj],
        "messages": [message_with_thinking, message_with_user_text],
        "metadata": metadata,
        "proxy_server_request": proxy_server_request,
    }

    sanitized = sanitize_claude_request_payload(payload)

    # Provider-facing payload must be completely unchanged
    assert payload["tools"][0] is tool_obj
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert payload["tools"][0]["description"] == "a" * 200
    content_0 = payload["messages"][0]["content"][0]
    content_1 = payload["messages"][1]["content"][0]
    assert isinstance(content_0, dict)
    assert isinstance(content_1, dict)
    assert content_0["signature"] == "opaque-signature"
    assert content_0 is thinking_block
    assert content_1["text"] == user_text
    assert content_1 is user_text_block
    assert payload["metadata"]["signature"] == "metadata-sig"
    assert payload["proxy_server_request"]["body"] == {"model": "claude-sonnet-4-6"}

    # Sanitized snapshot must have changes
    assert "input_schema" not in sanitized["tools"][0]
    assert "signature" not in sanitized["messages"][0]["content"][0]
    assert sanitized["metadata"] == metadata
    assert sanitized["proxy_server_request"]["body"] == {}


def test_sanitize_claude_request_payload_is_idempotent_for_truncated_fields():
    payload = {
        "system": "s" * 200,
        "tools": [
            {
                "name": "search",
                "description": "d" * 200,
                "input_schema": {"type": "object"},
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t" * 200,
                        "signature": "signature-to-remove",
                    },
                    {
                        "type": "tool_result",
                        "content": "r" * 200,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "u" * 5000,
                    }
                ],
            },
        ],
    }

    first = sanitize_claude_request_payload(payload)
    second = sanitize_claude_request_payload(first)

    assert second == first


def test_claude_request_payload_sanitizer_class_matches_public_function():
    payload = {
        "system": "s" * 200,
        "tools": [
            {
                "name": "search",
                "description": "d" * 200,
                "input_schema": {"type": "object"},
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t" * 200,
                        "signature": "signature-to-remove",
                    }
                ],
            }
        ],
    }

    assert ClaudeRequestPayloadSanitizer().sanitize(payload) == sanitize_claude_request_payload(payload)


def test_sanitize_response_payload_removes_provider_thinking_signatures():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "visible assistant response",
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "t" * 130,
                                "signature": "s" * 130,
                            }
                        ]
                    },
                }
            }
        ],
        "usage": {"total_tokens": 10},
    }

    sanitized = sanitize_response_payload(payload)

    assert sanitized == {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "visible assistant response",
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": f"{'t' * 100}... [+30 chars]",
                            }
                        ]
                    },
                }
            }
        ],
        "usage": {"total_tokens": 10},
    }
    assert payload["choices"][0]["message"]["provider_specific_fields"]["thinking_blocks"][0]["signature"] == "s" * 130


def test_claude_response_payload_sanitizer_class_matches_public_function():
    payload = {
        "choices": [
            {
                "message": {
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "t" * 130,
                                "signature": "s" * 130,
                            }
                        ]
                    }
                }
            }
        ]
    }

    assert ClaudeResponsePayloadSanitizer().sanitize(payload) == sanitize_response_payload(payload)


def test_openai_response_payload_sanitizer_class_matches_public_function():
    payload = {
        "object": "response",
        "instructions": "i" * 130,
        "tools": [
            {
                "name": "exec_command",
                "type": "function",
                "parameters": {"type": "object"},
            }
        ],
        "output": [{"type": "reasoning", "encrypted_content": "e" * 130}],
    }

    assert OpenAIResponsePayloadSanitizer().sanitize(payload) == sanitize_response_payload(payload)


def test_sanitize_response_payload_sanitizes_openai_response_shape():
    tool_parameters = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "schema description " + "s" * 130,
            }
        },
    }
    payload = {
        "object": "response",
        "instructions": "i" * 130,
        "tools": [
            {
                "name": "exec_command",
                "type": "function",
                "description": "tool description " + "d" * 130,
                "parameters": tool_parameters,
            },
            {
                "name": "apply_patch",
                "type": "custom",
                "format": {
                    "type": "grammar",
                    "definition": "grammar " + "g" * 130,
                },
            },
        ],
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": "r" * 130}],
                "encrypted_content": "e" * 130,
            },
            {
                "id": "msg_1",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "visible assistant response",
                    }
                ],
            },
        ],
    }

    sanitized = sanitize_response_payload(payload)

    assert sanitized == {
        "object": "response",
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [
            {
                "name": "exec_command",
                "type": "function",
                "description": f"tool description {'d' * 83}... [+47 chars]",
            },
            {
                "name": "apply_patch",
                "type": "custom",
                "format": {"type": "grammar"},
            },
        ],
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": f"{'r' * 100}... [+30 chars]"}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "visible assistant response",
                    }
                ],
            },
        ],
    }
    assert tool_parameters["properties"]["cmd"]["description"].startswith("schema description")
    assert payload["output"][0]["encrypted_content"] == "e" * 130


@pytest.mark.asyncio
async def test_async_logging_hook_sanitizes_response_without_changing_request_body():
    hook = RequestResponseSanitizerHook()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    kwargs = {"litellm_params": {"proxy_server_request": {"body": body}}}
    result = {
        "choices": [
            {
                "message": {
                    "provider_specific_fields": {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "reasoning",
                                "signature": "opaque-signature",
                            }
                        ]
                    }
                }
            }
        ]
    }

    updated_kwargs, updated_result = await hook.async_logging_hook(kwargs=kwargs, result=result, call_type="completion")

    assert updated_kwargs is kwargs
    assert updated_kwargs["litellm_params"]["proxy_server_request"]["body"] is body
    assert updated_result is result
    assert (
        result["choices"][0]["message"]["provider_specific_fields"]["thinking_blocks"][0]["signature"]
        == "opaque-signature"
    )


@pytest.mark.asyncio
async def test_async_logging_hook_sanitizes_standard_logging_object_response():
    hook = RequestResponseSanitizerHook()
    response = {
        "object": "response",
        "instructions": "i" * 130,
        "tools": [
            {
                "type": "function",
                "name": "search",
                "parameters": {"type": "object"},
            }
        ],
        "output": [{"type": "reasoning", "encrypted_content": "e" * 130}],
    }
    result = {"id": "resp_1"}
    kwargs = {"standard_logging_object": {"response": response}}

    updated_kwargs, updated_result = await hook.async_logging_hook(kwargs=kwargs, result=result, call_type="aresponses")

    assert updated_result is result
    assert updated_kwargs["standard_logging_object"]["response"] == {
        "object": "response",
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [{"type": "function", "name": "search"}],
        "output": [{"type": "reasoning"}],
    }
    assert response["tools"][0]["parameters"] == {"type": "object"}
    assert response["output"][0]["encrypted_content"] == "e" * 130


def test_logging_hook_sanitizes_standard_logging_object_response():
    hook = RequestResponseSanitizerHook()
    kwargs = {
        "standard_logging_object": {
            "response": {
                "object": "response",
                "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            }
        }
    }

    updated_kwargs, updated_result = hook.logging_hook(kwargs=kwargs, result={"id": "resp_1"}, call_type="responses")

    assert updated_result == {"id": "resp_1"}
    assert updated_kwargs["standard_logging_object"]["response"] == {
        "object": "response",
        "tools": [{"type": "function", "name": "search"}],
    }


def test_logging_hook_does_not_mutate_result_when_standard_logging_response_shares_object():
    hook = RequestResponseSanitizerHook()
    shared_response = {
        "object": "response",
        "instructions": "i" * 130,
        "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
        "output": [{"type": "reasoning", "encrypted_content": "e" * 130}],
    }
    kwargs = {"standard_logging_object": {"response": shared_response}}

    updated_kwargs, updated_result = hook.logging_hook(kwargs=kwargs, result=shared_response, call_type="responses")

    assert updated_result is shared_response
    assert shared_response["instructions"] == "i" * 130
    assert shared_response["tools"][0]["parameters"] == {"type": "object"}
    assert shared_response["output"][0]["encrypted_content"] == "e" * 130
    assert updated_kwargs["standard_logging_object"]["response"] == {
        "object": "response",
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [{"type": "function", "name": "search"}],
        "output": [{"type": "reasoning"}],
    }


class _CaptureStandardLoggingObject(CustomLogger):
    def __init__(self):
        super().__init__()
        self.standard_logging_object: dict[str, Any] | None = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_object = kwargs.get("standard_logging_object")


@pytest.mark.asyncio
async def test_responses_api_e2e_sanitizes_db_payload_without_changing_provider_response():
    capture_logger = _CaptureStandardLoggingObject()
    sanitizer_hook = RequestResponseSanitizerHook()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [sanitizer_hook, capture_logger]
    provider_payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": "i" * 130,
        "model": "gpt-5-mini",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e" * 130},
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible provider response", "annotations": []}],
            },
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "user": None,
        "metadata": {},
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=json.dumps(provider_payload).encode("utf-8"),
            request=httpx.Request(method="POST", url="https://api.openai.com/v1/responses"),
        )

    try:
        with patch.object(AsyncHTTPHandler, "post", new=mock_post):
            response = await litellm.aresponses(
                model="gpt-5-mini",
                input="hi",
                api_key="test-key",
            )
        for _ in range(50):
            if capture_logger.standard_logging_object is not None:
                break
            await asyncio.sleep(0.1)
    finally:
        litellm.callbacks = original_callbacks

    assert response.instructions == "i" * 130
    assert response.model_dump()["tools"][0]["parameters"] == {"type": "object"}
    assert response.model_dump()["output"][0]["encrypted_content"] == "e" * 130
    assert response.output_text == "visible provider response"

    assert capture_logger.standard_logging_object is not None
    stored_response = capture_logger.standard_logging_object["response"]
    assert stored_response["instructions"] == f"{'i' * 100}... [+30 chars]"
    assert stored_response["tools"][0]["type"] == "function"
    assert stored_response["tools"][0]["name"] == "search"
    assert "parameters" not in stored_response["tools"][0]
    assert "encrypted_content" not in stored_response["output"][0]


def test_sanitize_openai_request_payload_truncates_responses_request_fields(monkeypatch):
    monkeypatch.setenv("USER_TEXT_PREVIEW_MAX_LENGTH", "4000")
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "5000")
    payload = {
        "instructions": "i" * 130,
        "input": [
            {
                "role": "developer",
                "type": "message",
                "content": [{"type": "input_text", "text": "d" * 130}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<system-reminder>context</system-reminder>",
                    },
                    {"type": "input_text", "text": "u" * 4201},
                ],
            },
            {
                "role": "assistant",
                "type": "reasoning",
                "encrypted_content": "secret-reasoning",
                "summary": [{"type": "summary_text", "text": "r" * 130}],
            },
            {
                "role": "assistant",
                "type": "function_call",
                "name": "search",
                "arguments": "a" * 130,
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "tool description",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized == {
        "instructions": f"{'i' * 100}... [+30 chars]",
        "input": [
            {
                "role": "developer",
                "type": "message",
                "content": [{"type": "input_text", "text": f"{'d' * 100}... [+30 chars]"}],
            },
            {
                "role": "user",
                "type": "message",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<system-reminder>context</system-reminder>",
                    },
                    {"type": "input_text", "text": f"{'u' * 4000}... [+201 chars]"},
                ],
            },
            {
                "role": "assistant",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": f"{'r' * 100}... [+30 chars]"}],
            },
            {
                "role": "assistant",
                "type": "function_call",
                "name": "search",
                "arguments": f"{'a' * 100}... [+30 chars]",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "tool description",
            }
        ],
    }
    assert payload["input"][2]["encrypted_content"] == "secret-reasoning"
    assert payload["tools"][0]["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }


def test_sanitize_openai_request_payload_sanitizes_chat_completions_shape():
    payload = {
        "messages": [
            {"role": "system", "content": "s" * 130},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "a" * 130,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "o" * 130},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "d" * 130,
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    sanitized = OpenAIRequestPayloadSanitizer().sanitize(payload)

    assert sanitized == {
        "messages": [
            {"role": "system", "content": f"{'s' * 100}... [+30 chars]"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": f"{'a' * 100}... [+30 chars]",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": f"{'o' * 100}... [+30 chars]",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": f"{'d' * 100}... [+30 chars]",
                },
            }
        ],
    }


def test_sanitize_openai_request_payload_recursively_sanitizes_nested_tools():
    nested_tool_parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "schema description " + "s" * 130,
            }
        },
    }
    custom_tool_format = {
        "type": "grammar",
        "definition": "grammar " + "g" * 130,
    }
    payload = {
        "tools": [
            {
                "type": "mcp",
                "description": "parent " + "p" * 130,
                "tools": [
                    {
                        "name": "js",
                        "description": "nested " + "n" * 130,
                        "parameters": nested_tool_parameters,
                    }
                ],
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "format": custom_tool_format,
            },
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized == {
        "tools": [
            {
                "type": "mcp",
                "description": f"parent {'p' * 93}... [+37 chars]",
                "tools": [
                    {
                        "name": "js",
                        "description": f"nested {'n' * 93}... [+37 chars]",
                    }
                ],
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {
                    "type": "grammar",
                },
            },
        ]
    }
    assert nested_tool_parameters["properties"]["code"]["description"].startswith("schema description")
    assert custom_tool_format["definition"].startswith("grammar")


def test_sanitize_openai_request_payload_sanitizes_item_level_input_and_tools():
    payload = {
        "input": [
            {
                "name": "run_task",
                "type": "mcp_approval_request",
                "input": "i" * 130,
                "tools": [
                    {
                        "type": "mcp",
                        "tools": [
                            {
                                "name": "agent",
                                "description": "d" * 130,
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "agent_type": {
                                            "type": "string",
                                            "description": "p" * 130,
                                        }
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized == {
        "input": [
            {
                "name": "run_task",
                "type": "mcp_approval_request",
                "input": f"{'i' * 100}... [+30 chars]",
                "tools": [
                    {
                        "type": "mcp",
                        "tools": [
                            {
                                "name": "agent",
                                "description": f"{'d' * 100}... [+30 chars]",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    agent_type_schema = payload["input"][0]["tools"][0]["tools"][0]["parameters"]["properties"][
        "agent_type"
    ]
    assert agent_type_schema["description"] == "p" * 130


def test_sanitize_openai_request_payload_truncates_client_metadata_but_preserves_litellm_metadata():
    payload = {
        "input": "hi",
        "client_metadata": {"trace": "c" * 130},
        "litellm_metadata": {
            "headers": {"x-codex-turn-metadata": "h" * 130},
            "user_api_key_auth": "u" * 130,
        },
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized == {
        "input": "hi",
        "client_metadata": {"trace": f"{'c' * 100}... [+30 chars]"},
        "litellm_metadata": payload["litellm_metadata"],
    }


def test_sanitize_openai_request_payload_keeps_nested_proxy_body_empty():
    payload = {
        "model": "gpt-5.5",
        "input": "hi",
        "proxy_server_request": {
            "url": "http://example.com/v1/responses",
            "method": "POST",
            "body": {"model": "gpt-5.5", "input": "hi"},
        },
    }

    sanitized = sanitize_openai_request_payload(payload)

    assert sanitized["proxy_server_request"] == {
        "url": "http://example.com/v1/responses",
        "method": "POST",
        "body": {},
    }
    assert payload["proxy_server_request"]["body"] == {"model": "gpt-5.5", "input": "hi"}


@pytest.mark.asyncio
async def test_openai_async_pre_call_hook_sanitizes_proxy_server_request_body():
    hook = RequestResponseSanitizerHook()
    data = {
        "proxy_server_request": {
            "body": {
                "instructions": "i" * 130,
                "tools": [
                    {
                        "type": "function",
                        "name": "search",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        }
    }

    updated = await hook.async_pre_call_hook(
        user_api_key_dict=None,
        cache=None,
        data=data,
        call_type="responses",
    )

    assert updated is data
    assert data["proxy_server_request"]["body"] == {
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [{"type": "function", "name": "search"}],
    }


def test_openai_cli_writes_sanitized_json(tmp_path: Path):
    from litellm.integrations.request_response_sanitizer_hook import main

    input_path = tmp_path / "openai-request.json"
    output_path = tmp_path / "openai-request_sanitized.json"
    input_path.write_text(
        json.dumps(
            {
                "instructions": "i" * 130,
                "tools": [
                    {
                        "type": "function",
                        "name": "search",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main([str(input_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "instructions": f"{'i' * 100}... [+30 chars]",
        "tools": [{"type": "function", "name": "search"}],
    }


def test_openai_get_sanitized_output_path():
    assert get_sanitized_output_path(Path("openai-request.json")) == Path("openai-request_sanitized.json")
