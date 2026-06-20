# coding: utf-8
"""Pydantic 化节点公共基类 / Pydantic-config Node Base.

抽取 12 节点 __init__ 的 Union 校验样板 (~144 行重复) 到此处.
子类只需声明 ``ConfigSchema = XxxNodeConfig`` 即可:

>>> class FooNode(PydanticConfigNode):
...     ConfigSchema = FooNodeConfig
...     def _execute(self, input_data=None, **kw): ...

构造时:
- ``isinstance(config, ConfigSchema)``  → 直接使用
- ``dict / None``                       → ``ConfigSchema.model_validate``
- 其他类型                              → ``TypeError``
"""

from typing import Any, ClassVar, Optional, Union

from pydantic import BaseModel

from QuantNodes.core.node import BaseNode


class PydanticConfigNode(BaseNode):
    """统一处理 12 节点的 ``Union[dict, ConfigSchema, None]`` 配置入参.

    子类必须设置 ``ConfigSchema``; 构造完成后, 实例上挂载:

    - ``self.cfg`` : ``ConfigSchema`` 实例 (推荐访问入口)
    - ``self.config``: ``dict`` (BaseNode 协议要求)
    - ``self._xxx`` : 由 ``_ALIASES`` 声明的 cfg 字段别名 (向后兼容)

    自动 alias 规则 (Phase G3):
    声明 ``_ALIASES = {"_data_path": "data_path", ...}`` 后, 基类
    ``__init__`` 自动 copy cfg 字段到 self._xxx。list/tuple 字段会被浅拷贝
    以避免下游误改 cfg。
    """

    ConfigSchema: ClassVar[type[BaseModel]]
    _ALIASES: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        config: Union[dict, BaseModel, None] = None,
        **kwargs: Any,
    ) -> None:
        Schema = self.ConfigSchema
        if isinstance(config, Schema):
            cfg = config
            super().__init__(name, cfg.model_dump(), **kwargs)
        elif isinstance(config, dict) or config is None:
            cfg = Schema.model_validate(config or {})
            super().__init__(name, config, **kwargs)
        else:
            raise TypeError(
                f"config must be dict/None/{Schema.__name__}, "
                f"got {type(config).__name__}"
            )
        self.cfg: BaseModel = cfg
        # Auto-apply aliases
        for alias, field in self._ALIASES.items():
            value = getattr(self.cfg, field)
            if isinstance(value, list):
                value = list(value)  # shallow copy
            elif isinstance(value, tuple):
                value = list(value)
            setattr(self, alias, value)
