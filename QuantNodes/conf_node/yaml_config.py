# coding=utf-8
"""
YamlConfigNode - YAML 配置文件节点
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from QuantNodes.conf_node.base import ConfigNode


class YamlConfigNode(ConfigNode):
    """
    YAML 配置文件节点

    解析 .yaml/.yml 格式的配置文件。

    Examples:
        >>> node = YamlConfigNode(file_path="config.yaml")
        >>> config = node.execute()
        >>> node = YamlConfigNode(file_path="config.yaml", key="database")
        >>> db_config = node.execute()  # 只获取 database 下的配置
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        key: Optional[str] = None,
        encoding: str = 'utf-8',
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            file_path: YAML 文件路径
            key: 可选的顶层 key，只读取该 key 下的配置
            encoding: 文件编码，默认 utf-8
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(name=name, config=config, **kwargs)
        self.file_path = Path(file_path)
        self.key = key
        self.encoding = encoding

    def _get_config_path(self) -> Optional[Path]:
        return self.file_path

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.file_path}")

        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config. Install with: pip install pyyaml"
            )

        with open(self.file_path, 'r', encoding=self.encoding) as f:
            data = yaml.safe_load(f)

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ValueError(f"YAML file must contain a dictionary, got {type(data)}")

        if self.key:
            if self.key not in data:
                raise KeyError(f"Key '{self.key}' not found in {self.file_path}")
            return data[self.key]

        return data
