# coding=utf-8
"""
JSONConfigNode - JSON 配置文件节点
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from QuantNodes.conf_node.base import ConfigNode


class JSONConfigNode(ConfigNode):
    """
    JSON 配置文件节点

    解析 .json 格式的配置文件。

    Examples:
        >>> node = JSONConfigNode(file_path="config.json")
        >>> config = node.execute()
        >>> node = JSONConfigNode(file_path="config.json", key="database")
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
            file_path: JSON 文件路径
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
        """加载 JSON 配置"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding=self.encoding) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"JSON file must contain a dictionary, got {type(data)}")

        if self.key:
            if self.key not in data:
                raise KeyError(f"Key '{self.key}' not found in {self.file_path}")
            return data[self.key]

        return data
