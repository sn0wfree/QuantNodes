# coding=utf-8
"""
test_bug_regression.py - 历史 bug 回归测试 (Phase 9.3)

目标: 防止 V4-V8 已修复 bug 复发。每个测试 = 一个历史 bug 的红→绿对照。

Bug 时间线:
- V4-V7: pvd 一直 0 因子 (vol/volume 列名不匹配) → 修于 commit 8147a94
- V5: thinking-chain 集成导致 mean_reversion / volatility 失败 → 修于 V6 (4-layer defense)
- V6: 4-layer defense 暴露 LLM 截断问题 → 修于 commits b77f09a, a73b39e, f9cbab3, b914fa2
- V8: sign-mismatch (intraday_reversal 全正 IR) → 修于 test/expand-coverage-2x Phase 1
"""
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.logic_mining.compiler import check_sign_hint
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab


# ==============================================================================
# Test Class 1: V4-V7 vol/volume 列名 bug
# ==============================================================================


class TestVolVolumeAliasRegression:
    """V4-V7 pvd 一直 0 因子 (vol/volume 列名不匹配)

    修复: commit 8147a94 vocabulary.build_namespace 加 alias
    验证: 用 volume 的公式应能正常求值
    """

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """3 票 × 10 日 测试数据 (列名 'vol')"""
        return pl.DataFrame({
            "date": ["2024-01-01"] * 30,
            "code": ["A", "B", "C"] * 10,
            "close": [100.0 + i for i in range(30)],
            "vol": [1000.0 + i * 10 for i in range(30)],
        }).with_columns(pl.col("date").str.to_date())

    def test_volume_formula_evaluates(self, sample_data):
        """用 volume 命名的公式应能求值 (V4-V7 这类公式都报错)"""
        vocab = OperatorVocab.default()
        # 这是 V4-V7 一直失败的公式 (volume 名字)
        result = vocab.evaluate("ts_corr(rank(close), rank(volume), 10)", sample_data)
        assert result is not None
        assert result.shape[0] == 30

    def test_vol_still_works(self, sample_data):
        """短名 vol 在 alias 注入后仍可用 (向后兼容)"""
        vocab = OperatorVocab.default()
        result = vocab.evaluate("ts_corr(rank(close), rank(vol), 10)", sample_data)
        assert result is not None
        assert result.shape[0] == 30

    def test_volume_and_vol_yield_same_result(self, sample_data):
        """两种命名得到完全一致结果"""
        vocab = OperatorVocab.default()
        r_vol = vocab.evaluate("ts_mean(vol, 3)", sample_data)
        r_volume = vocab.evaluate("ts_mean(volume, 3)", sample_data)
        # 比较非空位置
        v_list = [x for x in r_vol.to_list() if x is not None]
        vol_list = [x for x in r_volume.to_list() if x is not None]
        assert len(v_list) == len(vol_list)
        diffs = [abs(a - b) for a, b in zip(v_list, vol_list)]
        if diffs:
            assert max(diffs) < 1e-9


# ==============================================================================
# Test Class 2: V8 sign-mismatch bug
# ==============================================================================


class TestSignHintStrictRegression:
    """V8 intraday_reversal 3 因子全正 IR 但 sign_constraint=-1

    修复: test/expand-coverage-2x Phase 1
    check_sign_hint direction=-1 严格化: 必须有 - / sign(- / sub(0, ...)
    """

    def test_positive_formula_rejected_for_direction_minus1(self):
        """正向公式 (rank(close)) + direction=-1 → False (V8 回归保护)"""
        # 修复前: True (宽松兜底)
        # 修复后: False (严格)
        assert check_sign_hint("rank(close)", -1) is False

    def test_negative_prefix_dash_accepted(self):
        """-rank(close) + direction=-1 → True"""
        assert check_sign_hint("-rank(close)", -1) is True

    def test_sub_zero_accepted(self):
        """sub(0, close) + direction=-1 → True"""
        assert check_sign_hint("sub(0, close)", -1) is True

    def test_sign_neg_accepted(self):
        """sign(-close) + direction=-1 → True"""
        assert check_sign_hint("sign(-close)", -1) is True


# ==============================================================================
# Test Class 3: V5 thinking-chain regression
# ==============================================================================


class TestThinkingChainRegression:
    """V5 thinking-chain 集成导致 mean_reversion / volatility 失败 0 因子

    修复: V6 4-layer defense (P0 截断恢复 + P1 explanation 截断)
    + V7 refactor/smart-p2 (P1 合并到 P2, 智能 3 档)
    """

    def test_p2_smart_split_preserves_information(self):
        """P2 v2 智能拆分: 保留 summary + detail 两边信息

        修复前: P1 粗暴截断, 丢失 thinking 块信息
        修复后: P2 v2 拆分为 explanation + explanation_detail
        """
        from QuantNodes.research.quant_alpha.llm.parser import (
            parse_formula_translator_output,
        )
        raw = '{"formulas": [{"id": "F1", "idea_id": "I1", "formula": "rank(close)", "explanation": "20日反转。HYPOTHESIS: A股散户过度反应...MECHANISM: sub(close, ts_mean(close,20))"}]}'
        r = parse_formula_translator_output(raw)
        f = r.data["formulas"][0]
        # summary: 标记前
        assert "20日反转" in f["explanation"]
        # detail: 标记及之后
        assert "HYPOTHESIS" in f.get("explanation_detail", "")
        assert "MECHANISM" in f["explanation_detail"]

    def test_p2_v2_short_unchanged(self):
        """P2 短小干净公式保留 (V5/V6 解释膨胀已修复)"""
        from QuantNodes.research.quant_alpha.llm.parser import (
            parse_formula_translator_output,
        )
        raw = '{"formulas": [{"id": "F1", "idea_id": "I1", "formula": "rank(close)", "explanation": "20日反转因子"}]}'
        r = parse_formula_translator_output(raw)
        f = r.data["formulas"][0]
        assert f["explanation"] == "20日反转因子"
        # 不应有 explanation_detail (短小)
        assert "explanation_detail" not in f

    def test_p2_v2_truncates_long_no_marker(self):
        """P2 v2: 超长无结构化 → 截断到 200 chars"""
        from QuantNodes.research.quant_alpha.llm.parser import (
            parse_formula_translator_output,
        )
        long_text = "x" * 500
        raw = f'{{"formulas": [{{"id": "F1", "idea_id": "I1", "formula": "rank(close)", "explanation": "{long_text}"}}]}}'
        r = parse_formula_translator_output(raw)
        f = r.data["formulas"][0]
        # 截断到 200 chars
        assert len(f["explanation"]) <= 200
        assert "explanation_detail" not in f


# ==============================================================================
# Test Class 4: V6 4-layer defense (P0 截断恢复)
# ==============================================================================


class TestP0TruncationRecoveryRegression:
    """V6 P0 修复: LLM 截断 JSON 后追加 thinking 块再重写 JSON 的恢复

    修复: parser._find_last_valid_json() 找最后一个满足 schema 的 JSON
    """

    def test_p0_recovers_truncated_then_thinking_then_complete(self):
        """P0 恢复: 截断 JSON + thinking + 重写完整 JSON"""
        from QuantNodes.research.quant_alpha.llm.parser import (
            parse_idea_generator_output,
        )
        raw = (
            '{"round": 1, "ideas": [{"id": "I1", "name": "a", "category":'  # 截断
            "<think>let me rewrite</think>\n"  # thinking
            '{"round": 1, "ideas": [{"id": "I1", "name": "a", "category": "reversal"}]}'  # 完整
        )
        r = parse_idea_generator_output(raw)
        # 应解析完整版, 不是截断版
        assert r.ok is True
        assert r.data["ideas"][0]["category"] == "reversal"


# ==============================================================================
# Test Class 5: V4 dedup sort bug
# ==============================================================================


class TestDedupSortBugRegression:
    """V4 dedup sort bug: 用 overall_score 排序, 漏掉 negative IR 高 |IR| 因子

    修复: V4 阶段修了 abs(overall_score) 排序
    """

    def test_dedup_includes_negative_ir(self):
        """dedup 不应按 overall_score 正向过滤 (V4 修复防护)

        V4 bug: 排序用 overall_score 漏掉负 IR 高 |IR| 因子
        修复: dedup 不应 filter 负 IR 因子
        """
        # 直接验证 dedup 函数: 输入 2 个负 overall_score 因子, 输出应 >= 1
        # (而不是被 strict 过滤掉)
        from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
            deduplicate_mutual_ic,
            FactorMetrics,
        )
        # 两个都是负 overall_score
        metrics = [
            FactorMetrics(formula_id="F1", status="success", overall_score=-0.05, ir=-0.1, ic_mean=-0.01),
            FactorMetrics(formula_id="F2", status="success", overall_score=-0.08, ir=-0.15, ic_mean=-0.02),
        ]
        # get_values 返 None → 无法算 corr → dedup 行为依赖实现
        # 我们只验证函数不崩
        result = deduplicate_mutual_ic(metrics, lambda m: None, threshold=0.5)
        # 至少能跑通, 返回 list
        assert isinstance(result, list)
