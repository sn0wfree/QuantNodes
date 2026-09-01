"""Demo: 运行归因模块, 生成示例报告."""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts/cmf')

import pandas as pd
from research_ml_anomaly_v3 import build_features, load_data, train_models
from explain_ml_anomaly import full_explain_pipeline


def main():
    print("=" * 70)
    print("ML异常归因 DEMO - 输出Markdown报告")
    print("=" * 70)

    df = build_features(load_data())['2020':'2025']
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    print("\n[1] 训练模型 (PIT全量)...")
    scaler, m_if, m_lof, m_zscore, t_if, t_lof, t_zscore = train_models(df, current_date=df.index[-1])
    print(f"    IF={t_if:.3f}s, LOF={t_lof:.3f}s, ZScore={t_zscore:.3f}s")

    print("\n[2] 运行归因流程 (R>60)...")
    explanations, meta = full_explain_pipeline(df, scaler, m_if, m_lof, m_zscore, r_threshold=60)
    print(f"    解释天数: {meta['n_explained']}")
    print(f"    ZScore分解: {meta['zscore_time']:.3f}s")
    print(f"    IF SHAP: {meta['if_shap_time']:.3f}s")
    print(f"    LOF近似: {meta['lof_approx_time']:.3f}s")
    print(f"    解释生成: {meta['explain_time']:.3f}s")
    print(f"    总耗时: {meta['total_time']:.2f}s")

    print("\n[3] 生成报告...")
    explanations_df = pd.DataFrame(explanations)
    explanations_df = explanations_df.sort_values("r_score", ascending=False)

    output_path = "reports/gold/ml_anomaly_explain_demo.md"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# ML异常归因报告 (Demo)\n\n")
        f.write(f"- 区间: {df.index[0].date()} ~ {df.index[-1].date()}\n")
        f.write(f"- 解释天数: {meta['n_explained']} (R>{meta['r_threshold']})\n")
        f.write(f"- 总耗时: {meta['total_time']:.2f}s\n\n")

        f.write("## Top10 高异常日\n\n")
        f.write("| 日期 | R分数 | IF | LOF | ZScore | 一致性 | Top1特征 | Top1 z | 解读 |\n")
        f.write("|------|-------|-----|------|--------|--------|----------|---------|------|\n")

        for _, row in explanations_df.head(10).iterrows():
            consensus_emoji = "✅" if row["consensus"] == "high" else "⚠️" if row["consensus"] == "disagree" else "🟢"
            f.write(f"| {row['date'].date()} | {row['r_score']:.1f} | {row['model_if']:.0f} | {row['model_lof']:.0f} | {row['model_zscore']:.0f} | {consensus_emoji} | {row['top1_feature']} | {row['top1_z']} | {row['interpretation']} |\n")

        f.write("\n## 经典案例\n\n")
        f.write("### 案例1: 3模型一致的高异常日 (取R最高)\n\n")
        high_consensus = explanations_df[explanations_df["consensus"] == "high"].head(3)
        for _, row in high_consensus.iterrows():
            f.write(f"**{row['date'].date()} (R={row['r_score']:.1f})**\n\n")
            f.write(f"- IF归因: {row['model_if']:.0f}/100\n")
            f.write(f"- LOF归因: {row['model_lof']:.0f}/100\n")
            f.write(f"- ZScore归因: {row['model_zscore']:.0f}/100\n")
            f.write(f"- Top1: {row['top1_feature']} ({row['top1_z']})\n")
            f.write(f"- Top2: {row['top2_feature']}\n")
            f.write(f"- Top3: {row['top3_feature']}\n\n")

        f.write("### 案例2: 模型分歧日 (取分歧中R最高的)\n\n")
        disagree = explanations_df[explanations_df["consensus"] == "disagree"].sort_values("r_score", ascending=False).head(3)
        if len(disagree) > 0:
            for _, row in disagree.iterrows():
                f.write(f"**{row['date'].date()} (R={row['r_score']:.1f})**\n\n")
                f.write(f"- IF: {row['model_if']:.0f} | LOF: {row['model_lof']:.0f} | ZScore: {row['model_zscore']:.0f}\n")
                f.write(f"- 解读: {row['interpretation']}\n\n")
        else:
            f.write("无分歧日\n\n")

        f.write("\n## 全部异常日列表\n\n")
        f.write(f"共{meta['n_explained']}天, 完整数据见 `reports/gold/ml_anomaly_explain_demo.csv`\n")

    explanations_df.to_csv(output_path.replace(".md", ".csv"), index=False)
    print(f"\n[4] 报告已生成:")
    print(f"    Markdown: {output_path}")
    print(f"    CSV: {output_path.replace('.md', '.csv')}")

    print("\n" + "=" * 70)
    print("Top5 高异常日预览:")
    print("=" * 70)
    for _, row in explanations_df.head(5).iterrows():
        print(f"\n{row['date'].date()} | R={row['r_score']:.1f} | [{row['consensus']}]")
        print(f"  IF={row['model_if']:.0f} | LOF={row['model_lof']:.0f} | ZScore={row['model_zscore']:.0f}")
        print(f"  {row['interpretation']}")


if __name__ == "__main__":
    main()