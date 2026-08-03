"""LLM 可解释性模块 — 通过 API 生成自然语言评估报告.

配置 (环境变量):
  RISK_LLM_ENDPOINT: API 端点 (默认: https://api.openai.com/v1)
  RISK_LLM_API_KEY:  API 密钥
  RISK_LLM_MODEL:    模型名 (默认: gpt-4o-mini)
"""
from __future__ import annotations
import os, json
from pathlib import Path


class LLMInterpreter:
    """Convert model evaluation details into natural language explanation.

    Uses OpenAI-compatible API. Falls back to template if unavailable.
    """

    def __init__(self):
        self.api_key = os.getenv("RISK_LLM_API_KEY")
        self.endpoint = os.getenv("RISK_LLM_ENDPOINT", "https://api.openai.com/v1")
        self.model = os.getenv("RISK_LLM_MODEL", "gpt-4o-mini")
        self.available = bool(self.api_key)

    def interpret(self, detail: dict) -> str:
        """Generate natural language explanation from model evaluation detail."""
        if not self.available:
            return self._fallback(detail)

        prompt = self._build_prompt(detail)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return self._fallback(detail) + f"\n\n(LLM unavailable: {e})"

    def _build_prompt(self, d: dict) -> str:
        weights = d.get("weights", {})
        uncertainty = d.get("weight_uncertainty", {})
        w_lines = "\n".join(
            f"    {k}: {v:.1%} (uncertainty={uncertainty.get(k, 0):.2f})"
            for k, v in weights.items()
        )

        return f"""你是量化波动率风险预警模型的解释器。请用中文分析以下模型状态, 输出三段话:

## 模型状态: {d.get('status', 'Unknown')} (健康评分: {d.get('health_score', 0)}/100)

### 性能指标
- 60d 滚动 AUC: {d.get('rolling_auc', 0):.3f} (阈值 0.55)
- 60d 滚动 Brier: {d.get('rolling_brier', 0):.3f} (阈值 0.30)
- ECE 校准误差: {d.get('rolling_ece', 0):.3f} (阈值 0.20)
- Climatology 差距: {d.get('auc_gap', 0):.3f} (阈值 -0.05)
- 高信号正报率: {d.get('high_signal_precision', 0):.1%}
- 高信号占比: {d.get('high_signal_ratio', 0):.1%}

### 权重分布
{w_lines}

### 建议动作
{d.get('action', 'continue')}

请输出:
1. 一句话总结当前状态和健康度
2. 主要问题或风险点 (如果有)
3. 建议操作"""

    def _fallback(self, d: dict) -> str:
        status = d.get("status", "Unknown")
        score = d.get("health_score", 0)
        weights = d.get("weights", {})

        lines = [
            f"状态: {status} ({score}/100)",
            "",
        ]
        if status == "Normal":
            lines.append("模型运行正常, 所有指标在阈值内, 无需操作。")
        elif status == "Warning":
            ece = d.get("rolling_ece", 0)
            lines.append(f"校准偏差: ECE={ece:.3f} 超过阈值 0.20, "
                         f"建议执行概率重校准 (Platt scaling)。")
        elif status == "Degraded":
            auc = d.get("rolling_auc", 0)
            lines.append(f"性能退化: 60d AUC={auc:.3f} 低于阈值 0.55, "
                         f"建议重建模型层。")
        else:
            lines.append("模型失效, 已自动切换到 Climatology 回退方案。")

        lines.append("")
        w_lines = ", ".join(f"{k}={v:.1%}" for k, v in weights.items())
        lines.append(f"当前权重: {w_lines}")
        lines.append(f"建议动作: {d.get('action', 'continue')}")
        return "\n".join(lines)