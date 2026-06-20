# coding=utf-8
"""
Web Fetch 工具

抓取指定 URL 的网页内容，支持 text/html/markdown 格式。
"""

import re
from typing import Any, Dict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .base import Tool


class WebFetchTool(Tool):
    """网页抓取工具

    抓取指定 URL 的内容，支持纯文本、HTML、Markdown 格式。
    """

    MAX_CONTENT = 50 * 1024  # 50KB
    TIMEOUT = 10.0
    USER_AGENT = "Mozilla/5.0 (compatible; QuantNodes/1.0)"

    def __init__(self) -> None:
        # H9 (2026-06-20): per-instance AsyncClient for connection pooling.
        # Previously each execute() opened a fresh client + paid a fresh
        # TCP+TLS handshake. With many URLs in one agent run, that's
        # multi-second savings.
        self._client = httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=True)

    async def aclose(self) -> None:
        """Close the underlying httpx client (call at end of session)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "抓取指定 URL 的网页内容。"
            "支持 text（提取纯文本）、html（返回原始 HTML）格式。"
            "禁止访问本地地址。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的 URL（必须是 http/https）",
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "html"],
                    "description": "返回格式：text 提取纯文本，html 返回原始 HTML",
                    "default": "text",
                },
            },
            "required": ["url"],
        }

    @property
    def read_only(self) -> bool:
        return True

    def _is_safe_url(self, url: str) -> bool:
        """检查 URL 是否安全（禁止本地地址）"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            hostname = parsed.hostname or ""
            if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return False
            if hostname.startswith("192.168.") or hostname.startswith("10."):
                return False
            return True
        except Exception:
            return False

    async def execute(self, url: str = "", format: str = "text", **kwargs: Any) -> Any:
        if not url:
            return {"error": "url is required"}
        if not self._is_safe_url(url):
            return {"error": f"URL not allowed: {url}"}

        try:
            resp = await self._client.get(url, headers={"User-Agent": self.USER_AGENT})
            resp.raise_for_status()
        except httpx.TimeoutException:
            return {"error": f"Request timed out ({self.TIMEOUT}s)"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

        content_type = resp.headers.get("content-type", "")
        html = resp.text

        if format == "html":
            text = html[:self.MAX_CONTENT]
            return {
                "url": url,
                "status_code": resp.status_code,
                "content_type": content_type,
                "content": text,
                "truncated": len(html) > self.MAX_CONTENT,
            }

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > self.MAX_CONTENT:
            text = text[:self.MAX_CONTENT] + "\n...[truncated]"

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        return {
            "url": url,
            "status_code": resp.status_code,
            "title": title,
            "content": text,
            "truncated": len(text) > self.MAX_CONTENT,
        }
