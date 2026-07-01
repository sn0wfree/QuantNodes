# coding=utf-8
"""Tests for ai/llm/openai.py — OpenAI client with retry, streaming, error handling.

Covers: creation, headers, API calls (mocked), error handling, retry logic,
streaming, Azure variant.
"""

from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.ai.llm.openai import OpenAIClient
from QuantNodes.ai.llm.base import Message, MessageRole, RateLimitError, AuthenticationError, APIError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    return OpenAIClient(api_key="test-key", base_url="https://api.test.com/v1")


@pytest.fixture
def mock_response():
    """Create a mock requests.Response for successful API call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    resp.raise_for_status = MagicMock()
    return resp


def _user_msg(text):
    return Message(role=MessageRole.USER, content=text)


# ============================================================================
# Creation
# ============================================================================

class TestOpenAIClientCreation:
    def test_creation_with_api_key(self):
        client = OpenAIClient(api_key="my-key")
        assert client is not None

    def test_creation_custom_params(self):
        client = OpenAIClient(
            api_key="key",
            base_url="https://custom.api.com/v1",
            model="gpt-4",
            timeout=30,
            max_retries=5,
        )
        assert client is not None

    def test_creation_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = OpenAIClient()
        assert client is not None

    def test_creation_empty_api_key_still_works(self):
        client = OpenAIClient(api_key="")
        assert client is not None


# ============================================================================
# Headers
# ============================================================================

class TestOpenAIClientHeaders:
    def test_bearer_token_header(self, client):
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"

    def test_content_type_header(self, client):
        headers = client._get_headers()
        assert "Content-Type" in headers

    def test_extra_headers_merged(self):
        client = OpenAIClient(
            api_key="key",
            extra_headers={"X-Custom": "value"},
        )
        headers = client._get_headers()
        assert headers["X-Custom"] == "value"


# ============================================================================
# API Calls (Mocked)
# ============================================================================

class TestOpenAIClientGenerate:
    def test_call_api_success(self, client, mock_response):
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client._call_api(
                messages=[_user_msg("Hi")],
                model="gpt-4",
                temperature=0.7,
            )
            assert result is not None
            mock_post.assert_called_once()

    def test_call_api_rate_limit_retries(self, client):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.raise_for_status.side_effect = Exception("429")
        rate_limit_resp.json.return_value = {"error": {"message": "rate limited"}}

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        ok_resp.raise_for_status = MagicMock()

        with patch("requests.post", side_effect=[rate_limit_resp, ok_resp]):
            with patch("time.sleep"):
                result = client._call_api(
                    messages=[_user_msg("Hi")],
                    model="gpt-4",
                )
                assert result is not None

    def test_call_api_auth_error(self, client):
        auth_resp = MagicMock()
        auth_resp.status_code = 401
        auth_resp.raise_for_status.side_effect = Exception("401")
        auth_resp.json.return_value = {"error": {"message": "unauthorized"}}

        with patch("requests.post", return_value=auth_resp):
            with pytest.raises(AuthenticationError):
                client._call_api(
                    messages=[_user_msg("Hi")],
                    model="gpt-4",
                )

    def test_call_api_server_error_retries(self, client):
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.raise_for_status.side_effect = Exception("500")
        error_resp.json.return_value = {"error": {"message": "server error"}}

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        ok_resp.raise_for_status = MagicMock()

        with patch("requests.post", side_effect=[error_resp, ok_resp]):
            with patch("time.sleep"):
                result = client._call_api(
                    messages=[_user_msg("Hi")],
                    model="gpt-4",
                )
                assert result is not None

    def test_call_api_timeout_retries(self, client):
        import requests as req_lib

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        ok_resp.raise_for_status = MagicMock()

        with patch("requests.post", side_effect=[req_lib.exceptions.Timeout(), ok_resp]):
            with patch("time.sleep"):
                result = client._call_api(
                    messages=[_user_msg("Hi")],
                    model="gpt-4",
                )
                assert result is not None


# ============================================================================
# Response Parsing
# ============================================================================

class TestOpenAIClientParsing:
    def test_parse_response(self, client):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = client._parse_response(data)
        assert result is not None
        assert result.content == "Hello!"

    def test_parse_stream_chunk(self, client):
        data = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }],
        }
        result = client._parse_stream_chunk(data)
        assert result is not None


# ============================================================================
# Model List
# ============================================================================

class TestOpenAIClientModels:
    def test_get_model_list(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "gpt-4"}, {"id": "gpt-3.5-turbo"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            models = client.get_model_list()
            assert "gpt-4" in models


# ============================================================================
# Azure OpenAI
# ============================================================================

class TestAzureOpenAI:
    def test_azure_creation(self):
        from QuantNodes.ai.llm.openai import AzureOpenAIClient
        client = AzureOpenAIClient(
            api_key="azure-key",
            azure_endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
        )
        assert client is not None

    def test_azure_uses_api_key_header(self):
        from QuantNodes.ai.llm.openai import AzureOpenAIClient
        client = AzureOpenAIClient(
            api_key="azure-key",
            azure_endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4",
        )
        headers = client._get_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "azure-key"
