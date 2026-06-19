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

    保留 ``self._xxx`` 实例属性的工作仍由子类完成 (向后兼容已有测试).
    """

    ConfigSchema: ClassVar[type[BaseModel]]

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
