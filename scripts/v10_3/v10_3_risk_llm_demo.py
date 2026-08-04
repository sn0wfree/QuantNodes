"""LLM 可解释性演示 — 通过 API 生成自然语言评估报告.

需要环境变量:
  export RISK_LLM_API_KEY=your_api_key
  export RISK_LLM_ENDPOINT=https://api.openai.com/v1 (或兼容端点)
  export RISK_LLM_MODEL=gpt-4o-mini (或 qwen-plus 等)

如果未配置, 自动使用模板回退。
"""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    AutoRiskWarning, load_etf_data,
)
from QuantNodes.strategy.momentum_etf_rotation.risk.llm import LLMInterpreter
log = print


def main():
    log("=" * 60)
    log("LLM 可解释性演示")
    log("=" * 60)

    etf = load_etf_data()

    # 更新一次获取最新状态
    aw = AutoRiskWarning(etf, "equity")
    detail = aw.weekly_update()

    log(f"\n模型状态: {detail['status']} ({detail['health_score']}/100)")
    log(f"权重: {', '.join(f'{k}={v:.1%}' for k, v in detail['weights'].items())}")
    log(f"建议动作: {detail['action']}")
    log(f"\n--- LLM 解释 ---")

    llm = LLMInterpreter()
    explanation = llm.interpret(detail)
    log(f"\n{explanation}")

    if not llm.available:
        log(f"\n提示: 设置 RISK_LLM_API_KEY 环境变量可启用 LLM 解释。")
        log(f"当前使用模板回退。")

    log("\n[Done]")


if __name__ == "__main__":
    main()