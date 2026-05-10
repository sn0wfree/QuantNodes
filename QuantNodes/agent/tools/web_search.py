# coding=utf-8
"""
Web Search 工具

使用 DuckDuckGo HTML 端点执行网络搜索。
"""

from typing import Any, Dict, List
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from .base import Tool


class WebSearchTool(Tool):
    """网络搜索工具

    使用 DuckDuckGo HTML 端点搜索关键词，返回搜索结果。
    """

    TIMEOUT = 10.0
    USER_AGENT = "Mozilla/5.0 (compatible; QuantNodes/1.0)"
    SEARCH_URL = "https://html.duckduckgo.com/html/"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "网络搜索工具：使用 DuckDuckGo 搜索关键词。"
            "返回标题、URL、摘要。无需 API Key。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数（默认 5，最大 20）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str = "", max_results: int = 5, **kwargs: Any) -> Any:
        if not query:
            return {"error": "query is required"}

        max_results = min(max(max_results, 1), 20)

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(
                    self.SEARCH_URL,
                    params={"q": query},
                    headers={"User-Agent": self.USER_AGENT},
                )
                resp.raise_for_status()
        except httpx.TimeoutException:
            return {"error": f"Search timed out ({self.TIMEOUT}s)"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

        soup = BeautifulSoup(resp.text, "lxml")
        results: List[Dict[str, str]] = []

        for result_div in soup.select(".result"):
            if len(results) >= max_results:
                break

            title_tag = result_div.select_one(".result__title a")
            snippet_tag = result_div.select_one(".result__snippet")

            if not title_tag:
                continue

            href = title_tag.get("href", "")
            title = title_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            results.append({
                "title": title,
                "url": href,
                "snippet": snippet,
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }
