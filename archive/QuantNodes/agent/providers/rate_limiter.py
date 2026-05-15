# coding=utf-8
"""
速率限制模块 (Rate Limiter)

提供 Token Bucket 算法实现，用于控制 LLM API 请求频率。
解决 OpenRouter 免费账号的严格速率限制问题（每天 50 次请求）。
"""

import time
import asyncio
import threading
from typing import Optional


class TokenBucket:
    """同步令牌桶速率限制器

    用于多线程环境下的请求频率控制。

    Args:
        requests_per_second: 每秒允许的请求数（免费账号建议 0.5）
        burst: 突发容量，允许的最大突发请求数
    """

    def __init__(self, requests_per_second: float = 0.5, burst: int = 1):
        """
        Args:
            requests_per_second: 每秒允许的请求数，默认 0.5 即每2秒1次请求
            burst: 突发容量，默认 1 表示每次最多处理1个请求
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")

        self.rate = requests_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, blocking: bool = True) -> bool:
        """获取令牌

        Args:
            blocking: 是否阻塞等待令牌可用

        Returns:
            True if token acquired, False if not (non-blocking mode only)
        """
        with self.lock:
            self._refill()

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            if not blocking:
                return False

            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
            return True

    def try_acquire(self) -> bool:
        """非阻塞尝试获取令牌"""
        return self.acquire(blocking=False)

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数"""
        with self.lock:
            self._refill()
            return self.tokens


class AsyncTokenBucket:
    """异步令牌桶速率限制器

    用于 asyncio 环境下的请求频率控制。

    Args:
        requests_per_second: 每秒允许的请求数（免费账号建议 0.5）
        burst: 突发容量
    """

    def __init__(self, requests_per_second: float = 0.5, burst: int = 1):
        """
        Args:
            requests_per_second: 每秒允许的请求数，默认 0.5 即每2秒1次请求
            burst: 突发容量，默认 1
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")

        self.rate = requests_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def _refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self) -> None:
        """获取令牌，必要时等待

        这是阻塞方法，会等待直到令牌可用。
        """
        async with self.lock:
            await self._refill()

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0

    async def try_acquire(self) -> bool:
        """非阻塞尝试获取令牌

        Returns:
            True if token acquired, False otherwise
        """
        async with self.lock:
            await self._refill()

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

    async def wait_time(self) -> float:
        """计算获取令牌需要等待的时间

        Returns:
            等待时间（秒），如果立即可用则返回 0
        """
        async with self.lock:
            await self._refill()

            if self.tokens >= 1:
                return 0.0

            return (1 - self.tokens) / self.rate

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数（非线程安全，仅供调试）"""
        return self.tokens


class SlidingWindowRateLimiter:
    """滑动窗口速率限制器

    另一种速率限制实现，在固定时间窗口内限制请求数。

    Args:
        max_requests: 时间窗口内最大请求数
        window_seconds: 时间窗口长度（秒）
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []
        self.lock = threading.Lock()

    def _clean_old_requests(self) -> None:
        """清除过期请求记录"""
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def acquire(self, blocking: bool = True) -> bool:
        """获取许可"""
        with self.lock:
            self._clean_old_requests()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return True

            if not blocking:
                return False

            oldest = self.requests[0]
            wait_time = oldest + self.window_seconds - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            self._clean_old_requests()
            self.requests.append(time.time())
            return True

    def try_acquire(self) -> bool:
        """非阻塞尝试获取许可"""
        return self.acquire(blocking=False)


class AsyncSlidingWindowRateLimiter:
    """异步滑动窗口速率限制器"""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []
        self.lock = asyncio.Lock()

    async def _clean_old_requests(self) -> None:
        """清除过期请求记录"""
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    async def acquire(self) -> None:
        """获取许可"""
        async with self.lock:
            await self._clean_old_requests()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return

            oldest = self.requests[0]
            wait_time = oldest + self.window_seconds - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            await self._clean_old_requests()
            self.requests.append(time.time())

    async def try_acquire(self) -> bool:
        """非阻塞尝试获取许可"""
        async with self.lock:
            await self._clean_old_requests()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return True

            return False


class AdaptiveRateLimiter:
    """自适应速率限制器

    根据 API 响应自动调整请求速率。
    当检测到限流错误时自动降低速率，正常时逐步提升。

    Args:
        initial_rps: 初始速率（每秒请求数）
        min_rps: 最小速率
        max_rps: 最大速率
        increase_factor: 正常时速率增加因子
        decrease_factor: 触发限流时速率降低因子
    """

    def __init__(
        self,
        initial_rps: float = 0.5,
        min_rps: float = 0.1,
        max_rps: float = 2.0,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
    ):
        self.current_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.last_adjust_time = time.time()
        self.lock = threading.Lock()

        self._bucket = TokenBucket(initial_rps, burst=1)

    def report_success(self) -> None:
        """报告成功调用，适当提高速率"""
        with self.lock:
            now = time.time()
            if now - self.last_adjust_time >= 1.0:
                self.current_rps = min(self.max_rps, self.current_rps * self.increase_factor)
                self._bucket = TokenBucket(self.current_rps, burst=1)
                self.last_adjust_time = now

    def report_rate_limit(self) -> None:
        """报告限流错误，大幅降低速率"""
        with self.lock:
            self.current_rps = max(self.min_rps, self.current_rps * self.decrease_factor)
            self._bucket = TokenBucket(self.current_rps, burst=1)

    def report_server_error(self) -> None:
        """报告服务器错误，中等降低速率"""
        with self.lock:
            self.current_rps = max(self.min_rps, self.current_rps * 0.7)
            self._bucket = TokenBucket(self.current_rps, burst=1)

    def acquire(self, blocking: bool = True) -> bool:
        """获取令牌"""
        return self._bucket.acquire(blocking)

    @property
    def current_rate(self) -> float:
        """当前速率"""
        return self.current_rps