# coding=utf-8
"""
Configuration Node Module

提供配置节点体系，用于从各种来源加载配置：
- IniConfigNode: INI 配置文件
- YamlConfigNode: YAML 配置文件
- JSONConfigNode: JSON 配置文件
- EnvConfigNode: 环境变量

Examples:
    >>> # 从 INI 文件读取
    >>> ini = IniConfigNode(file_path="config.ini", section="database")
    >>> settings = ini.execute()
    >>>
    >>> # 从 YAML 文件读取
    >>> yaml_config = YamlConfigNode(file_path="config.yaml", key="app")
    >>> app_config = yaml_config.execute()
    >>>
    >>> # 从环境变量读取
    >>> env = EnvConfigNode(prefix="MYAPP_", types={'PORT': int})
    >>> env_config = env.execute()
"""

from QuantNodes.conf_node.base import ConfigNode
from QuantNodes.conf_node.ini_config import IniConfigNode
from QuantNodes.conf_node.yaml_config import YamlConfigNode
from QuantNodes.conf_node.env_config import EnvConfigNode
from QuantNodes.conf_node.json_config import JSONConfigNode

__all__ = [
    'ConfigNode',
    'IniConfigNode',
    'YamlConfigNode',
    'EnvConfigNode',
    'JSONConfigNode',
]
