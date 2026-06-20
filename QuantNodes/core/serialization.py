# coding=utf-8
"""
序列化工具模块 - 提供多种序列化方案

方案对比：
| 方案         | 安全 | 可读 | 速度 | 跨语言 | 零依赖 | 推荐场景               |
|-------------|------|------|------|--------|--------|------------------------|
| JSON        | ✅   | ✅   | ⭐⭐  | ✅     | ✅     | 默认、调试、跨系统交互  |
| JSON+zlib   | ✅   | ⚠️   | ⭐⭐⭐| ✅     | ✅     | 网络传输、持久化存储    |
| msgpack     | ✅   | ❌   | ⭐⭐⭐⭐ | ✅     | ❌     | 高性能场景（可选依赖）  |
| pickle      | ⚠️   | ❌   | ⭐⭐⭐⭐ | ❌     | ✅     | 仅限可信内部环境        |

使用建议：
- 开发/调试 -> serialize_json()
- 网络传输 -> serialize_compact()
- 极致性能 -> serialize_msgpack() (需要 msgpack)
"""

from __future__ import annotations

import json
import zlib
import warnings
from typing import Union, Optional

from QuantNodes.core.expression import Expression


# ============================================================================
# 序列化方案
# ============================================================================

def serialize_json(expr, indent: Optional[int] = 2) -> str:
    """
    JSON 序列化 - 默认方案

    特点：安全、可读、跨语言、零依赖
    """
    return json.dumps(expr.serialize(), indent=indent, ensure_ascii=False)


def serialize_json_bytes(expr: Expression) -> bytes:
    """JSON 序列化（字节形式）"""
    return serialize_json(expr, indent=None).encode('utf-8')


def serialize_compact(expr, compress_level: int = 6) -> bytes:
    """
    紧凑序列化 - 推荐用于网络传输

    JSON + zlib 压缩，零额外依赖
    压缩率通常可达 30%-50%
    """
    json_bytes = serialize_json_bytes(expr)
    return zlib.compress(json_bytes, level=compress_level)


def serialize_msgpack(expr) -> bytes:
    """
    msgpack 序列化 - 极致性能

    要求：pip install msgpack
    特点：比 JSON 小 30-40%，速度快 2-3 倍
    """
    try:
        import msgpack
    except ImportError:
        raise ImportError(
            "msgpack 未安装。请运行: pip install msgpack"
        ) from None
    return msgpack.packb(expr.serialize(), use_bin_type=True)


def serialize_pickle(expr: Expression, protocol: int = 4) -> bytes:
    """
    Pickle 序列化 - 仅限内部可信环境

    警告：存在代码执行安全风险，仅反序列化可信来源的数据！
    """
    import pickle
    warnings.warn(
        "Pickle 序列化存在安全风险，仅在可信内部环境使用。"
        "建议优先使用 serialize_compact() 或 serialize_json()。",
        UserWarning,
        stacklevel=2
    )
    return pickle.dumps(expr, protocol=protocol)


# ============================================================================
# 反序列化方案
# ============================================================================

def deserialize_json(data: Union[str, bytes]) -> Expression:
    """从 JSON 反序列化"""
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    expr_data = json.loads(data)
    return Expression.from_dict(expr_data)


def deserialize_compact(data: bytes) -> Expression:
    """从紧凑格式（JSON + zlib）反序列化"""
    json_bytes = zlib.decompress(data)
    return deserialize_json(json_bytes)


def deserialize_msgpack(data: bytes) -> Expression:
    """从 msgpack 反序列化"""
    try:
        import msgpack
    except ImportError:
        raise ImportError(
            "msgpack 未安装。请运行: pip install msgpack"
        ) from None
    expr_data = msgpack.unpackb(data, raw=False)
    return Expression.deserialize(expr_data)


def deserialize_pickle(data: bytes) -> Expression:
    """
    从 pickle 反序列化

    警告：仅反序列化可信来源的数据！
    """
    import pickle
    warnings.warn(
        "Pickle 反序列化存在安全风险，仅反序列化可信来源的数据。",
        UserWarning,
        stacklevel=2
    )
    return pickle.loads(data)


# ============================================================================
# 自动检测反序列化
# ============================================================================

def deserialize_auto(data: Union[str, bytes]) -> Expression:
    """
    自动检测格式并反序列化

    支持：JSON(str/bytes), JSON+zlib(bytes), msgpack(bytes), pickle(bytes)
    """
    if isinstance(data, str):
        return deserialize_json(data)

    if not isinstance(data, bytes):
        raise TypeError(f"不支持的数据类型: {type(data)}")

    # 尝试 zlib 压缩格式
    if len(data) >= 2 and data[0:2] == b'x\x9c':  # zlib 魔数
        try:
            return deserialize_compact(data)
        except Exception:
            pass

    # 尝试 msgpack (第一个字节通常是 map 标记 0x80-0x8f)
    if data and 0x80 <= data[0] <= 0x8f:
        try:
            return deserialize_msgpack(data)
        except Exception:
            pass

    # 尝试 JSON
    try:
        return deserialize_json(data)
    except Exception:
        pass

    # 最后尝试 pickle（带警告）
    return deserialize_pickle(data)


# ============================================================================
# Protobuf 序列化 - 高性能跨语言方案
# ============================================================================

_PROTOBUF_AVAILABLE = False
try:
    import importlib.util
    _PROTOBUF_AVAILABLE = importlib.util.find_spec("google.protobuf") is not None
except ImportError:
    pass


# TODO [未完成]: 动态生成表达式的 protobuf schema
#
# 设计思路：
# - 当前实现：使用 struct_pb2.Struct 进行通用序列化（见 serialize_proto）
# - 预期目标：大规模场景下生成专属 .proto 文件以提升性能
#
# 完整实现需要：
# 1. 定义 Expression 的 protobuf 消息结构
# 2. 动态生成 .proto 描述符
# 3. 使用 protoc 编译或动态编译
#
# 当前状态：stub - 只有 pass，无实际功能
# 优先级：低 - 当前实现已满足需求
def _get_proto_descriptor():
    """
    动态生成表达式的 protobuf schema

    说明：不需要预编译 .proto 文件，动态构建描述符
    """
    # 简单实现：先序列化为 dict 再用 protobuf 打包
    # 完整实现需要生成 .proto 文件并编译
    pass


def serialize_proto(expr) -> bytes:
    """
    Protobuf 序列化

    要求：pip install protobuf
    特点：高性能、跨语言、强类型、前向兼容
    """
    if not _PROTOBUF_AVAILABLE:
        raise ImportError(
            "protobuf 未安装。请运行: pip install protobuf"
        )

    from google.protobuf import struct_pb2

    data = expr.serialize()
    struct = struct_pb2.Struct()
    struct.update(data)
    return struct.SerializeToString()


def _fix_numbers(obj):
    """递归修复数字类型：Protobuf 会把整数转成浮点数，转换回来"""
    if isinstance(obj, dict):
        return {k: _fix_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_numbers(v) for v in obj]
    elif isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def deserialize_proto(data: bytes) -> Expression:
    """从 Protobuf 反序列化"""
    if not _PROTOBUF_AVAILABLE:
        raise ImportError(
            "protobuf 未安装。请运行: pip install protobuf"
        )

    from google.protobuf import struct_pb2

    struct = struct_pb2.Struct()
    struct.ParseFromString(data)

    # 转换为 dict
    from google.protobuf.json_format import MessageToDict
    data_dict = MessageToDict(struct, preserving_proto_field_name=True)

    # 修复被转成浮点数的整数
    data_dict = _fix_numbers(data_dict)
    return Expression.deserialize(data_dict)


# ============================================================================
# 加密序列化 - 用于敏感策略保护
# ============================================================================

def serialize_encrypted(
    expr: Expression,
    key: Union[str, bytes],
    algorithm: str = 'AES-GCM'
) -> bytes:
    """
    加密序列化 - 保护敏感策略表达式

    Args:
        expr: 表达式对象
        key: 加密密钥（字符串或 bytes）
        algorithm: 加密算法（默认 AES-GCM，带认证）

    要求：使用 Python 标准库 cryptography
          pip install cryptography

    安全特性：
    ✅  AES-256-GCM 加密
    ✅  带认证标签，防止篡改
    ✅  随机 nonce，相同明文每次加密结果不同
    ✅  密钥派生（PBKDF2），支持字符串密码
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise ImportError(
            "cryptography 未安装。请运行: pip install cryptography"
        ) from None

    import os

    # 1. 先序列化为紧凑 JSON
    plaintext = expr.to_json(indent=None).encode('utf-8')

    # 2. 处理密钥
    if isinstance(key, str):
        # 字符串密码 -> 使用 PBKDF2 派生密钥
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key_bytes = kdf.derive(key.encode('utf-8'))
        key_type = b'P'  # PBKDF2 派生的密钥
    else:
        # 直接使用 bytes 密钥
        key_bytes = key
        salt = b''
        key_type = b'R'  # 原始密钥

    # 3. AES-GCM 加密
    nonce = os.urandom(12)  # 96 bits = NIST 推荐
    aesgcm = AESGCM(key_bytes)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # 4. 打包：[key_type(1)] + [salt_len(1)] + [salt] + [nonce] + [ciphertext]
    # 格式说明：
    # - key_type: 'P' = 密码派生, 'R' = 原始密钥
    # - salt_len: salt 的长度（原始密钥时为 0）
    # - salt: PBKDF2 的盐值（16 字节）
    # - nonce: AES-GCM 的 nonce（12 字节）
    # - ciphertext: 加密数据 + 16 字节认证标签
    salt_len = len(salt)
    return b''.join([
        key_type,
        bytes([salt_len]),
        salt,
        nonce,
        ciphertext,
    ])


def deserialize_encrypted(
    data: bytes,
    key: Union[str, bytes]
) -> Expression:
    """
    解密反序列化

    Args:
        data: 加密的字节数据
        key: 解密密钥（字符串或 bytes）

    Raises:
        cryptography.exceptions.InvalidTag: 密钥错误或数据被篡改
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise ImportError(
            "cryptography 未安装。请运行: pip install cryptography"
        ) from None

    # 1. 解包
    key_type = data[0:1]
    salt_len = data[1]
    offset = 2

    salt = data[offset:offset + salt_len]
    offset += salt_len

    nonce = data[offset:offset + 12]
    offset += 12

    ciphertext = data[offset:]

    # 2. 处理密钥
    if key_type == b'P':
        # PBKDF2 密钥派生
        if not isinstance(key, str):
            raise ValueError("数据使用字符串密码加密，请提供字符串密钥")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key_bytes = kdf.derive(key.encode('utf-8'))
    else:
        # 原始密钥
        if isinstance(key, str):
            raise ValueError("数据使用原始密钥加密，请提供 bytes 密钥")
        key_bytes = key

    # 3. AES-GCM 解密（自动验证认证标签）
    aesgcm = AESGCM(key_bytes)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    # 4. 反序列化
    return Expression.deserialize(plaintext)


# ============================================================================
# 节点序列化方案
# ============================================================================

def serialize_node_json(node, indent: Optional[int] = 2) -> str:
    """
    节点 JSON 序列化

    Args:
        node: BaseNode 实例
        indent: 缩进空格数，None 表示紧凑格式

    Returns:
        JSON 字符串
    """
    return json.dumps(node.serialize(), indent=indent, ensure_ascii=False)


def serialize_node_json_bytes(node) -> bytes:
    """节点 JSON 序列化（字节形式）"""
    return serialize_node_json(node, indent=None).encode('utf-8')


def serialize_node_compact(node, compress_level: int = 6) -> bytes:
    """
    节点紧凑序列化 - 推荐用于网络传输/持久化

    JSON + zlib 压缩，零额外依赖
    压缩率通常可达 30%-50%
    """
    json_bytes = serialize_node_json_bytes(node)
    return zlib.compress(json_bytes, level=compress_level)


def serialize_node_msgpack(node) -> bytes:
    """
    节点 msgpack 序列化 - 极致性能

    要求：pip install msgpack
    """
    try:
        import msgpack
    except ImportError:
        raise ImportError(
            "msgpack 未安装。请运行: pip install msgpack"
        ) from None
    return msgpack.packb(node.serialize(), use_bin_type=True)


# ============================================================================
# 节点反序列化方案
# ============================================================================

def deserialize_node_json(data: Union[str, bytes]):
    """
    从 JSON 反序列化节点

    Args:
        data: JSON 字符串或字节

    Returns:
        BaseNode 实例
    """
    from QuantNodes.core.node import BaseNode

    if isinstance(data, bytes):
        data = data.decode('utf-8')
    node_data = json.loads(data)
    return BaseNode.deserialize(node_data)


def deserialize_node_compact(data: bytes):
    """
    从紧凑格式（JSON + zlib）反序列化节点

    Args:
        data: 压缩的字节数据

    Returns:
        BaseNode 实例
    """

    json_bytes = zlib.decompress(data)
    return deserialize_node_json(json_bytes)


def deserialize_node_msgpack(data: bytes):
    """
    从 msgpack 反序列化节点

    Args:
        data: msgpack 字节数据

    Returns:
        BaseNode 实例
    """
    from QuantNodes.core.node import BaseNode

    try:
        import msgpack
    except ImportError:
        raise ImportError(
            "msgpack 未安装。请运行: pip install msgpack"
        ) from None
    node_data = msgpack.unpackb(data, raw=False)
    return BaseNode.deserialize(node_data)


# ============================================================================
# 节点自动检测反序列化
# ============================================================================

def deserialize_node_auto(data: Union[str, bytes]):
    """
    自动检测格式并反序列化节点

    支持：JSON(str/bytes), JSON+zlib(bytes), msgpack(bytes)
    """
    if isinstance(data, str):
        return deserialize_node_json(data)

    if not isinstance(data, bytes):
        raise TypeError(f"不支持的数据类型: {type(data)}")

    # 尝试 zlib 压缩格式
    if len(data) >= 2 and data[0:2] == b'x\x9c':
        try:
            return deserialize_node_compact(data)
        except Exception:
            pass

    # 尝试 msgpack (第一个字节通常是 map 标记 0x80-0x8f)
    if data and 0x80 <= data[0] <= 0x8f:
        try:
            return deserialize_node_msgpack(data)
        except Exception:
            pass

    # 尝试 JSON
    return deserialize_node_json(data)


# ============================================================================
# 节点加密序列化
# ============================================================================

def serialize_node_encrypted(node, key: Union[str, bytes]) -> bytes:
    """
    节点加密序列化 - 保护敏感策略

    Args:
        node: BaseNode 实例
        key: 加密密钥（字符串或 bytes）

    要求：pip install cryptography
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise ImportError(
            "cryptography 未安装。请运行: pip install cryptography"
        ) from None

    import os

    # 1. 先序列化为紧凑 JSON
    plaintext = node.serialize()
    plaintext_bytes = json.dumps(plaintext, separators=(',', ':')).encode('utf-8')

    # 2. 处理密钥
    if isinstance(key, str):
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key_bytes = kdf.derive(key.encode('utf-8'))
        key_type = b'P'
    else:
        key_bytes = key
        salt = b''
        key_type = b'R'

    # 3. AES-GCM 加密
    nonce = os.urandom(12)
    aesgcm = AESGCM(key_bytes)
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)

    # 4. 打包
    salt_len = len(salt)
    return b''.join([
        key_type,
        bytes([salt_len]),
        salt,
        nonce,
        ciphertext,
    ])


def deserialize_node_encrypted(data: bytes, key: Union[str, bytes]):
    """
    解密反序列化节点

    Args:
        data: 加密的字节数据
        key: 解密密钥
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        raise ImportError(
            "cryptography 未安装。请运行: pip install cryptography"
        ) from None

    # 1. 解包
    key_type = data[0:1]
    salt_len = data[1]
    offset = 2

    salt = data[offset:offset + salt_len]
    offset += salt_len

    nonce = data[offset:offset + 12]
    offset += 12

    ciphertext = data[offset:]

    # 2. 处理密钥
    if key_type == b'P':
        if not isinstance(key, str):
            raise ValueError("数据使用字符串密码加密，请提供字符串密钥")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key_bytes = kdf.derive(key.encode('utf-8'))
    else:
        if isinstance(key, str):
            raise ValueError("数据使用原始密钥加密，请提供 bytes 密钥")
        key_bytes = key

    # 3. AES-GCM 解密
    aesgcm = AESGCM(key_bytes)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    # 4. 反序列化
    return deserialize_node_json(plaintext)


# 注：原 _update_expression_serialization() 已删除 (Phase F, 2026-06-20)。
# 该函数原本通过 monkey patching 给 Expression 类注入 to_proto/from_proto 等方法,
# 已被 expression.py 中的直接实现取代 (expression.py:159-179), 无外部 caller。
# 详见 git log: 历史 commit 9b8a64c 引入后, 2026-05 改写为直接方法。
