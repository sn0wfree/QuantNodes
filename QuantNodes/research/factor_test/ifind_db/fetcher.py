# coding: utf-8
"""iFinD API 封装 + Markdown 表格解析 + 限流 + 本地缓存"""

import json
import time
import hashlib
import sys
from pathlib import Path

import pandas as pd

# ── iFinD API 配置 ──────────────────────────────────────────────
IFIND_SKILL_DIR = Path.home() / '.agents/skills/ifind'
IFIND_CONFIG = IFIND_SKILL_DIR / 'mcp_config.json'

# 将 skill 目录加入 sys.path 以导入 call.py
if str(IFIND_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(IFIND_SKILL_DIR))


def _load_auth_token() -> str:
    """从 iFinD MCP 配置读取 auth_token"""
    if not IFIND_CONFIG.exists():
        raise FileNotFoundError(
            f"iFinD 配置不存在: {IFIND_CONFIG}\n"
            "请先配置密钥: MCP官网 -> 个人中心 -> 密钥"
        )
    cfg = json.loads(IFIND_CONFIG.read_text(encoding='utf-8'))
    token = cfg.get('auth_token', '')
    if not token:
        raise ValueError("iFinD auth_token 为空，请重新配置")
    return token


class IFindFetcher:
    """iFinD API 调用 + Markdown 解析 + 限流 + 缓存"""

    # H17: 限流默认 0.5s (免费版 2 QPS), 付费版可设小
    DEFAULT_RATE_LIMIT_S = 0.5
    # H17: 缓存默认 7 天 (604800 秒)
    DEFAULT_CACHE_TTL_S = 7 * 86400

    def __init__(
        self,
        cache_dir: str | Path = None,
        rate_limit_s: float | None = None,
        cache_ttl_s: int | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path(__file__).parent / 'cache'
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_time = 0.0
        # H17: 限流 / 缓存 TTL 全部可覆盖
        self.rate_limit_s = (
            rate_limit_s if rate_limit_s is not None else self.DEFAULT_RATE_LIMIT_S
        )
        self.cache_ttl_s = (
            cache_ttl_s if cache_ttl_s is not None else self.DEFAULT_CACHE_TTL_S
        )

        # 延迟加载 call 函数 (避免导入时阻塞)
        self._call_fn = None
        self._auth_token = _load_auth_token()

    def _get_call_fn(self):
        """延迟导入 iFinD call 函数"""
        if self._call_fn is None:
            from call import call as _call
            self._call_fn = _call
        return self._call_fn

    def _rate_limit(self):
        """限流: 每次调用间隔至少 self.rate_limit_s。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_call_time = time.time()

    def _cache_key(self, server_type: str, tool_name: str, params: dict) -> str:
        """生成缓存键"""
        raw = f"{server_type}:{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        """从本地缓存加载 (TTL 由 self.cache_ttl_s 决定)。"""
        path = self._cache_dir / f"{key}.parquet"
        if path.exists():
            if (time.time() - path.stat().st_mtime) < self.cache_ttl_s:
                return pd.read_parquet(path)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        """保存到本地缓存"""
        path = self._cache_dir / f"{key}.parquet"
        df.to_parquet(path, index=False)

    def query(self, server_type: str, tool_name: str, params: dict) -> pd.DataFrame:
        """调用 iFinD API, 返回 DataFrame"""
        cache_key = self._cache_key(server_type, tool_name, params)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        call_fn = self._get_call_fn()
        result = call_fn(server_type, tool_name, params)

        if not result.get('ok'):
            raise RuntimeError(
                f"iFinD API 错误 [{server_type}/{tool_name}]: "
                f"{result.get('error', result)}"
            )

        df = self._parse_response(result)
        if df is not None and not df.empty:
            self._save_cache(cache_key, df)
        return df

    def _parse_response(self, result: dict) -> pd.DataFrame:
        """解析 iFinD JSON-RPC 响应, 提取 Markdown 表格"""
        data = result.get('data', {})
        content = data.get('result', {}).get('content', [])
        if not content:
            return pd.DataFrame()

        text = content[0].get('text', '')
        if not text:
            return pd.DataFrame()

        return self._parse_markdown_table(text)

    def _parse_markdown_table(self, text: str) -> pd.DataFrame:
        """解析 Markdown 表格为 DataFrame"""
        lines = text.strip().split('\n')
        # 筛选 | 开头的行
        table_lines = [l for l in lines if l.strip().startswith('|')]
        if len(table_lines) < 2:
            return pd.DataFrame()

        # 解析 header
        header = [c.strip() for c in table_lines[0].split('|')[1:-1]]
        # 跳过 separator 行 (|---|---|...)
        data_lines = [l for l in table_lines[1:] if not l.strip().replace('|', '').replace('-', '').replace(' ', '') == '']

        rows = []
        for line in data_lines:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if len(cells) == len(header):
                rows.append(cells)

        if not rows:
            return pd.DataFrame(columns=header)

        df = pd.DataFrame(rows, columns=header)
        # 清理数字列
        for col in df.columns:
            df[col] = self._try_convert_numeric(df[col])
        return df

    def _try_convert_numeric(self, series: pd.Series) -> pd.Series:
        """尝试将字符串列转为数值"""
        # 移除千分位逗号、万亿/亿/万 等中文单位
        cleaned = series.astype(str).str.replace(',', '', regex=False)
        # 处理中文单位
        multiplier = pd.Series(1.0, index=series.index)
        mask_wan = cleaned.str.contains('万', na=False)
        multiplier[mask_wan] = 1e4
        cleaned = cleaned.str.replace('万', '', regex=False)
        mask_yi = cleaned.str.contains('亿', na=False)
        multiplier[mask_yi] = 1e8
        cleaned = cleaned.str.replace('亿', '', regex=False)
        mask_wan_yi = cleaned.str.contains('万亿', na=False)
        multiplier[mask_wan_yi] = 1e12
        cleaned = cleaned.str.replace('万亿', '', regex=False)

        try:
            numeric = pd.to_numeric(cleaned, errors='coerce') * multiplier
            # 如果大部分能转成功, 就转
            if numeric.notna().sum() > len(series) * 0.5:
                return numeric
        except Exception:
            pass
        return series


class IFindFetcherStub:
    """测试用 stub - 不调用真实 API, 返回预设数据"""

    def __init__(self):
        self._responses = {}
        self._calls = []

    def register(self, server_type: str, tool_name: str, params: dict,
                 response_df: pd.DataFrame):
        """注册预设响应"""
        key = (server_type, tool_name, json.dumps(params, sort_keys=True))
        self._responses[key] = response_df

    def query(self, server_type: str, tool_name: str, params: dict) -> pd.DataFrame:
        """返回预设数据"""
        key = (server_type, tool_name, json.dumps(params, sort_keys=True))
        self._calls.append((server_type, tool_name, params))
        if key in self._responses:
            return self._responses[key].copy()
        return pd.DataFrame()

    @property
    def calls(self):
        return list(self._calls)
