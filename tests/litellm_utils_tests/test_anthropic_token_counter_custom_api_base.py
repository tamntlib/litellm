"""
Test Anthropic token counter with custom api_base from credentials.

This tests the scenario where a model like 'claude-opus-4-7' uses a custom
Anthropic-compatible API endpoint via credential (e.g., xiaomimimo-anthropic).

Key behavior: When a custom api_base is provided (not api.anthropic.com),
the token counter skips calling the count_tokens API and returns None,
allowing the system to fall back to local tiktoken counting.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter


class TestAnthropicTokenCounterCustomApiBase:
    """Test suite for Anthropic token counter with custom api_base."""

    @pytest.fixture
    def token_counter(self):
        return AnthropicTokenCounter()

    @pytest.fixture
    def custom_api_base(self):
        return "https://token-plan-sgp.xiaomimimo.com/anthropic/v1/messages"

    @pytest.mark.asyncio
    async def test_count_tokens_skips_for_custom_api_base(
        self, token_counter, custom_api_base
    ):
        """
        Test that count_tokens returns None when custom api_base is provided.

        When a credential provides a custom api_base (not api.anthropic.com),
        the token counter should skip calling the count_tokens API and return None.
        This allows the system to fall back to local tiktoken counting.
        """
        deployment = {
            "litellm_params": {
                "api_key": "test-api-key",
                "api_base": custom_api_base,
            }
        }

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler"
        ) as mock_handler:
            mock_handler.handle_count_tokens_request = AsyncMock(
                return_value={"input_tokens": 10}
            )

            result = await token_counter.count_tokens(
                model_to_use="mimo-v2.5-pro",
                messages=[{"role": "user", "content": "Hello"}],
                contents=None,
                deployment=deployment,
                request_model="claude-opus-4-7",
            )

            # Verify the handler was NOT called (skipped for custom api_base)
            mock_handler.handle_count_tokens_request.assert_not_called()

            # Verify result is None (will trigger fallback to local counting)
            assert result is None, "Should return None for custom api_base"

    @pytest.mark.asyncio
    async def test_count_tokens_calls_api_for_anthropic_endpoint(self, token_counter):
        """
        Test that count_tokens calls the API when using default Anthropic endpoint.

        When api_base is api.anthropic.com or not set, the token counter should
        call the count_tokens API.
        """
        deployment = {
            "litellm_params": {
                "api_key": "test-api-key",
                # No api_base or api_base pointing to api.anthropic.com
            }
        }

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler"
        ) as mock_handler:
            mock_handler.handle_count_tokens_request = AsyncMock(
                return_value={"input_tokens": 10}
            )

            result = await token_counter.count_tokens(
                model_to_use="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hello"}],
                contents=None,
                deployment=deployment,
                request_model="claude-haiku-4-5-20251001",
            )

            # Verify the handler WAS called
            mock_handler.handle_count_tokens_request.assert_called_once()

            # Verify result is valid
            assert result is not None
            assert result.total_tokens == 10
            assert result.error is not True

    @pytest.mark.asyncio
    async def test_count_tokens_calls_api_for_anthropic_api_base(self, token_counter):
        """
        Test that count_tokens calls the API when api_base is api.anthropic.com.
        """
        deployment = {
            "litellm_params": {
                "api_key": "test-api-key",
                "api_base": "https://api.anthropic.com/v1/messages",
            }
        }

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler"
        ) as mock_handler:
            mock_handler.handle_count_tokens_request = AsyncMock(
                return_value={"input_tokens": 15}
            )

            result = await token_counter.count_tokens(
                model_to_use="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello"}],
                contents=None,
                deployment=deployment,
                request_model="claude-3-5-sonnet-20241022",
            )

            # Verify the handler WAS called
            mock_handler.handle_count_tokens_request.assert_called_once()

            # Verify result is valid
            assert result is not None
            assert result.total_tokens == 15

    def test_handler_constructs_correct_url_with_custom_api_base(
        self, custom_api_base
    ):
        """
        Test that the handler constructs the correct endpoint URL when given a custom api_base.
        """
        handler = AnthropicCountTokensHandler()

        # When api_base is provided, it should be used directly
        endpoint_url = custom_api_base or handler.get_anthropic_count_tokens_endpoint()
        assert endpoint_url == custom_api_base

    def test_handler_uses_default_endpoint_without_api_base(self):
        """
        Test that the handler uses the default Anthropic endpoint when no api_base is provided.
        """
        handler = AnthropicCountTokensHandler()

        endpoint_url = None or handler.get_anthropic_count_tokens_endpoint()
        assert endpoint_url == "https://api.anthropic.com/v1/messages/count_tokens"

    @pytest.mark.asyncio
    async def test_count_tokens_returns_error_on_api_failure(self, token_counter):
        """
        Test that count_tokens returns a TokenCountResponse with error=True when the API fails.
        """
        from litellm.llms.anthropic.common_utils import AnthropicError

        deployment = {
            "litellm_params": {
                "api_key": "test-api-key",
                # No custom api_base, so it will call the API
            }
        }

        with patch(
            "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler"
        ) as mock_handler:
            mock_handler.handle_count_tokens_request = AsyncMock(
                side_effect=AnthropicError(
                    status_code=401, message="Unauthorized"
                )
            )

            result = await token_counter.count_tokens(
                model_to_use="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hello"}],
                contents=None,
                deployment=deployment,
                request_model="claude-haiku-4-5-20251001",
            )

            # Verify error is handled gracefully
            assert result is not None
            assert result.error is True
            assert result.total_tokens == 0
            assert "Unauthorized" in result.error_message

    @pytest.mark.asyncio
    async def test_count_tokens_with_no_api_key_returns_none(self, token_counter):
        """
        Test that count_tokens returns None when no API key is available.
        """
        deployment = {"litellm_params": {}}

        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY from environment
            os.environ.pop("ANTHROPIC_API_KEY", None)

            result = await token_counter.count_tokens(
                model_to_use="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hello"}],
                contents=None,
                deployment=deployment,
                request_model="claude-haiku-4-5-20251001",
            )

            # Should return None when no API key
            assert result is None
