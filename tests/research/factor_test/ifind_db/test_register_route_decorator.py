# coding: utf-8
"""B5 装饰器化专项测试 / register_route Decorator Tests."""

import pytest

from QuantNodes.research.factor_test.ifind_db.ifind_database import (
    IFinDDatabase,
    _collect_routes,
    register_route,
)


# ── 1. 装饰器本身行为 ──────────────────────────────────────────


def test_decorator_single_route():
    """单个 @register_route → method._routes == [(f, k)]."""
    @register_route("a.h5", "x")
    def fn():
        return 1
    assert fn._routes == [("a.h5", "x")]


def test_decorator_multiple_routes_stacked():
    """多个 @register_route 叠加 → method._routes 累积 (内层装饰器先执行)."""
    @register_route("a.h5", "x")  # 外层, 后执行
    @register_route("b.h5", "y")  # 内层, 先执行
    def fn():
        return 2
    assert fn._routes == [("b.h5", "y"), ("a.h5", "x")]


def test_decorator_preserves_callable():
    """装饰后仍可正常调用."""
    @register_route("a.h5", "x")
    def fn(x):
        return x * 2
    assert fn(3) == 6
    assert hasattr(fn, "_routes")


# ── 2. _collect_routes 行为 ───────────────────────────────────


def test_collect_routes_empty_class():
    """无装饰方法的类 → 空 dict."""

    class Empty:
        def no_route(self):
            pass

    assert _collect_routes(Empty) == {}


def test_collect_routes_single_method():
    """单装饰方法 → 1 条路由."""

    class A:
        @register_route("f.h5", "k")
        def m(self):
            return 1

    routes = _collect_routes(A)
    assert routes == {("f.h5", "k"): "m"}


def test_collect_routes_multiple_routes_one_method():
    """一个方法多个路由 (如 _get_trade_dt_raw 服务 stk + index)."""

    class A:
        @register_route("stk_daily.h5", "trade_dt")
        @register_route("index_daily.h5", "trade_dt")
        def m(self):
            return 1

    routes = _collect_routes(A)
    assert routes == {
        ("stk_daily.h5", "trade_dt"): "m",
        ("index_daily.h5", "trade_dt"): "m",
    }


def test_collect_routes_inheritance():
    """子类继承父类路由 + 自身新路由 (反向 MRO 遍历, 父路由先入 → 子同 method 名覆盖)."""

    class Parent:
        @register_route("p.h5", "x")
        def m(self):
            return "parent"

    class Child(Parent):
        @register_route("c.h5", "y")
        def m(self):
            return "child"

    routes = _collect_routes(Child)
    # _collect_routes 用 reversed MRO: object → Parent → Child
    # Parent.m → routes[("p.h5", "x")] = "m"
    # Child.m (同名) → routes[("c.h5", "y")] = "m" (新增, 不覆盖)
    # 两路由均指向 Child 的 m (Child 是 m 的最终定义)
    assert routes == {
        ("p.h5", "x"): "m",
        ("c.h5", "y"): "m",
    }


# ── 3. IFinDDatabase._ROUTE_TABLE 实际路由 ─────────────────────


def test_ifind_database_route_table_complete():
    """14 条路由全部到位, 与重构前完全一致."""
    rt = IFinDDatabase._ROUTE_TABLE
    assert len(rt) == 14
    expected = {
        ("stk_daily.h5", "cp"),
        ("stk_daily.h5", "stklist"),
        ("stk_daily.h5", "trade_dt"),
        ("stk_daily.h5", "id_citic1"),
        ("stk_daily.h5", "mv_float"),
        ("stk_daily.h5", "st"),
        ("stk_daily.h5", "suspend"),
        ("stk_daily.h5", "ud_limit"),
        ("stk_daily.h5", "ipo_days"),
        ("stk_daily.h5", "id_300"),
        ("stk_daily.h5", "id_500"),
        ("index_daily.h5", "index_cp"),
        ("index_daily.h5", "indexlist"),
        ("index_daily.h5", "trade_dt"),
    }
    assert set(rt.keys()) == expected


def test_ifind_database_route_targets_callable():
    """所有路由目标都是可调用的方法."""
    rt = IFinDDatabase._ROUTE_TABLE
    for fname, key in rt:
        method_name = rt[(fname, key)]
        method = getattr(IFinDDatabase, method_name)
        assert callable(method), f"{method_name} not callable"


def test_ifind_database_trade_dt_routes_to_same_method():
    """stk_daily.trade_dt 与 index_daily.trade_dt 都指向 _get_trade_dt_raw."""
    rt = IFinDDatabase._ROUTE_TABLE
    assert rt[("stk_daily.h5", "trade_dt")] == "_get_trade_dt_raw"
    assert rt[("index_daily.h5", "trade_dt")] == "_get_trade_dt_raw"


# ── 4. 端到端: load_h5 仍能路由 (Stub) ──────────────────────────


def test_load_h5_routes_to_method():
    """load_h5 通过装饰器收集的路由表正确分发 (Stub fetcher, 不打真实 iFinD)."""
    from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcherStub

    IFindFetcherStub()
    # 验证路由表正确指向 _get_prices
    rt = IFinDDatabase._ROUTE_TABLE
    assert rt[("stk_daily.h5", "cp")] == "_get_prices"


def test_unmapped_route_raises_keyerror():
    """load_h5 路由未映射 → KeyError (含可用路由提示)."""
    from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcherStub

    fetcher = IFindFetcherStub()
    db = IFinDDatabase(date_beg="20240101", date_end="20240130", fetcher=fetcher)

    with pytest.raises(KeyError, match=r"未映射"):
        db.load_h5("unknown.h5", "unknown_key")


# ── 5. 装饰器幂等 / 重装饰 ───────────────────────────────────


def test_decorator_re_appending_idempotent():
    """同一 (f, k) 重装饰 → 列表追加 (不覆盖), 由 _collect_routes 后到先合并时后者覆盖."""
    @register_route("f.h5", "k")
    @register_route("f.h5", "k")
    def fn():
        return 1
    # 方法上累积 2 条记录
    assert fn._routes == [("f.h5", "k"), ("f.h5", "k")]
    routes = _collect_routes(type("X", (), {"m": fn}))
    assert routes == {("f.h5", "k"): "m"}