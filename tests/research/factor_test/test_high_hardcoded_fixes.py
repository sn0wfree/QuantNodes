# coding: utf-8
"""Verify that remaining HIGH-severity hardcoded issues are fixed.

H5: _INDUSTRY_MAP 30 申万行业 hardcoded → industry_map constructor arg
H6: INDEX_MAPPING/INDEX_CP_MAPPING hardcoded → JSON override + resolve_index_mapping
H7: factor_score_node 3 * 29 * group magic number → configurable n_size * n_ind * group
H8: factor_score_node group = 5 hardcoded → ScoreSetting.n_quantile_groups
H9: LoadDataNode fallback data path → data_path required Pydantic validation
"""
import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.ifind_db.ifind_database import IFinDDatabase
from QuantNodes.research.factor_test.utils.constants import (
    INDEX_MAPPING, INDEX_CP_MAPPING,
    resolve_index_mapping,
)
from QuantNodes.research.factor_test.config import ScoreSetting
from QuantNodes.research.factor_test.nodes.configs import LoadDataNodeConfig
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode


class TestH5IndustryMapConfigurable:
    """H5: IFinDDatabase industry_map constructor parameter"""

    def test_default_30_sw_industries(self):
        """默认 30 申万一级行业"""
        db = IFinDDatabase(date_beg='20250101', date_end='20251231')
        assert len(db._industry_map) == 30
        assert db._industry_map['银行'] == 17

    def test_custom_industry_map(self):
        """自定义行业映射可覆盖"""
        custom = {'银行': 1, '地产': 2}
        db = IFinDDatabase(date_beg='20250101', date_end='20251231',
                           industry_map=custom)
        assert db._industry_map == custom
        assert len(db._industry_map) == 2


class TestH6IndexMappingOverride:
    """H6: INDEX_MAPPING via JSON override"""

    def test_default_2_indices(self):
        """默认 HS300 + ZZ500"""
        assert len(INDEX_MAPPING) == 2
        assert 'HS300' in INDEX_MAPPING
        assert 'ZZ500' in INDEX_MAPPING

    def test_default_cp_mapping(self):
        """默认指数代码映射"""
        assert len(INDEX_CP_MAPPING) == 2
        assert INDEX_CP_MAPPING['HS300'] == '000300.SH'
        assert INDEX_CP_MAPPING['ZZ500'] == '000905.SH'

    def test_custom_index_mapping_merge(self):
        """resolve_index_mapping 合并默认 + 自定义"""
        custom = {'MY_IDX': ('custom.h5', 'id_my')}
        merged = resolve_index_mapping({'INDEX_MAPPING': custom})
        assert merged['MY_IDX'] == ('custom.h5', 'id_my')
        # 默认保留
        assert merged['HS300'] == INDEX_MAPPING['HS300']


class TestH7H8ScoreConfigurable:
    """H7/H8: factor_score_node 魔数 3*29*group → 可配置"""

    def test_score_setting_defaults(self):
        """ScoreSetting 默认值: 29 行业, 3 市值组, 5 分位"""
        s = ScoreSetting()
        assert s.n_industries == 29
        assert s.n_size_groups == 3
        assert s.n_quantile_groups == 5

    def test_score_setting_custom(self):
        """自定义行业数 / 市值组数 / 分位数"""
        s = ScoreSetting(n_industries=30, n_size_groups=5, n_quantile_groups=10)
        assert s.n_industries == 30
        assert s.n_size_groups == 5
        assert s.n_quantile_groups == 10

    def test_score_node_uses_config(self):
        """FactorScoreNode 读取 config 值, 不用硬编码"""
        node = FactorScoreNode(config={
            'n_industries': 30,
            'n_size_groups': 5,
            'n_quantile_groups': 10,
        })
        assert node._n_industries == 30
        assert node._n_size_groups == 5
        assert node._n_quantile_groups == 10
        # min_count = 5 * 30 * 10 = 1500
        assert node._n_size_groups * node._n_industries * node._n_quantile_groups == 1500


class TestH9LoadDataRequiredPath:
    """H9: LoadDataNode 不再有硬编码 fallback 路径"""

    def test_missing_data_path_raises(self):
        """缺 data_path → ValidationError"""
        with pytest.raises(ValidationError):
            LoadDataNodeConfig()

    def test_none_data_path_raises(self):
        """None → ValidationError"""
        with pytest.raises(ValidationError):
            LoadDataNodeConfig(data_path=None)

    def test_empty_string_raises_on_execute(self):
        """空字符串 → _execute ValueError"""
        node = LoadDataNode(config={'data_path': ''})
        with pytest.raises(ValueError, match="data_path required"):
            node._execute()
