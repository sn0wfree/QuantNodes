# coding=utf-8
"""
IniConfigNode - INI 配置文件节点
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union

from QuantNodes.conf_node.base import ConfigNode


class IniConfigNode(ConfigNode):
    """
    INI 配置文件节点

    解析 .ini 格式的配置文件，支持多 section。

    Examples:
        >>> # 读取指定 section
        >>> node = IniConfigNode(file_path="config.ini", section="database")
        >>> db_config = node.execute()
        >>> print(db_config['host'])
        >>>
        >>> # 读取所有 section
        >>> node = IniConfigNode(file_path="config.ini")
        >>> all_config = node.execute()  # 返回 {'section1': {...}, 'section2': {...}}
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        section: Optional[str] = None,
        encoding: str = 'utf-8',
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            file_path: INI 文件路径
            section: 要读取的 section，None 表示读取所有 section
            encoding: 文件编码，默认 utf-8
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(name=name, config=config, **kwargs)
        self.file_path = Path(file_path)
        self.section = section
        self.encoding = encoding

    def _get_config_path(self) -> Optional[Path]:
        return self.file_path

    def _load_config(self) -> Dict[str, Any]:
        """加载 INI 配置"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"INI file not found: {self.file_path}")

        parser = configparser.ConfigParser()
        parser.read(self.file_path, encoding=self.encoding)

        if self.section:
            if self.section not in parser:
                raise KeyError(f"Section '{self.section}' not found in {self.file_path}")
            return dict(parser.items(self.section))
        else:
            return {s: dict(parser.items(s)) for s in parser.sections()}
