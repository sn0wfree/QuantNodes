# coding: utf-8
"""Node 12: 汇总报告 / Factor Test Report Node

Migrated from factor_output.py:539-616 save_testresult()
Output: Parquet/JSON instead of xlwings Excel
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Union

import pandas as pd
import numpy as np

from QuantNodes.core.node import BaseNode

logger = logging.getLogger(__name__)
from QuantNodes.research.factor_test.nodes.configs import ReportNodeConfig


class FactorTestReportNode(BaseNode):
    """汇总所有分析结果, 输出到文件

    输入: context 中所有分析结果
    输出: FactorTestReport (dict)
    """

    def __init__(self, name: str = "FactorTestReport",
                 config: Union[dict, ReportNodeConfig, None] = None, **kwargs):
        # T0-4: 预先 Union 化
        if isinstance(config, ReportNodeConfig):
            cfg = config
            super().__init__(name, cfg.model_dump(), **kwargs)
        elif isinstance(config, dict) or config is None:
            cfg = ReportNodeConfig.model_validate(config or {})
            super().__init__(name, config, **kwargs)
        else:
            raise TypeError(
                f"config must be dict/None/ReportNodeConfig, got {type(config).__name__}"
            )
        # P-1: 路径优先级 env QUANTNODES_OUTPUT_DIR > expanduser > default
        self._output_dir = self._resolve_output_dir(cfg.dir)
        self._output_format = list(cfg.format)

    @staticmethod
    def _resolve_output_dir(default: str) -> Path:
        """P-1: env 变量优先, 然后 expanduser, 最后 default"""
        env = os.environ.get("QUANTNODES_OUTPUT_DIR")
        if env:
            return Path(env).expanduser()
        return Path(default).expanduser()

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})

        report = {
            'factor_name': context.get('LoadData', {}).get('factor', pd.DataFrame()).columns[0]
                if hasattr(context.get('LoadData', {}).get('factor', pd.DataFrame()), 'columns')
                else 'unknown',
            'timestamp': datetime.now().isoformat(),
        }

        # 收集各分析结果
        if 'ICAnalyzer' in context:
            ic = context['ICAnalyzer']
            report['ic'] = {
                'ic_mean': float(ic['ic_result'].get('IC均值', np.nan)),
                'ic_std': float(ic['ic_result'].get('IC标准差', np.nan)),
                'icir': float(ic['ic_result'].get('ICIR', np.nan)),
                'rank_ic_mean': float(ic['rank_ic_result'].get('rankIC均值', np.nan)),
                'rank_icir': float(ic['rank_ic_result'].get('rankICIR', np.nan)),
            }

        if 'GroupAnalyzer' in context:
            ga = context['GroupAnalyzer']
            report['group'] = {
                'n_groups': ga.get('n_groups', 0),
                'group_eva_exc': ga.get('group_eva_exc', pd.DataFrame()).to_dict()
                    if isinstance(ga.get('group_eva_exc'), pd.DataFrame) else {},
            }

        if 'LongShort' in context:
            ls = context['LongShort']
            if isinstance(ls.get('eva_total'), pd.DataFrame):
                report['longshort'] = {
                    'eva': ls['eva_total'].to_dict(),
                }

        if 'FactorScore' in context and context['FactorScore']:
            sc = context['FactorScore']
            report['score'] = {
                'eva': sc.get('eva', pd.DataFrame()).to_dict()
                    if isinstance(sc.get('eva'), pd.DataFrame) else {},
            }

        if 'RiskCorrelation' in context:
            rc = context['RiskCorrelation']
            report['risk_corr'] = {
                'mean': rc.get('mean', pd.DataFrame()).to_dict()
                    if isinstance(rc.get('mean'), pd.DataFrame) else {},
            }

        # 输出到文件
        self._save_report(report, context)

        return report

    def _save_report(self, report, context):
        """保存报告到文件"""
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        factor_name = report.get('factor_name', 'factor')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for fmt in self._output_format:
            if fmt == 'json':
                import json
                # 转换不可序列化的对象
                def default_serializer(obj):
                    if isinstance(obj, (pd.Series, pd.DataFrame)):
                        return obj.to_dict()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return str(obj)

                path = output_dir / f"factor_test_{factor_name}_{timestamp}.json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=default_serializer)
                logger.info(f"报告已保存: {path}")

            elif fmt == 'parquet':
                # 保存各分析结果为 parquet
                for key, value in context.items():
                    if key.startswith('_'):
                        continue
                    if isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, pd.DataFrame) and not sub_val.empty:
                                path = output_dir / f"{factor_name}_{key}_{sub_key}.parquet"
                                sub_val.to_parquet(path)
                    elif isinstance(value, pd.DataFrame) and not value.empty:
                        path = output_dir / f"{factor_name}_{key}.parquet"
                        value.to_parquet(path)
                logger.info(f"Parquet 文件已保存至: {output_dir}")
