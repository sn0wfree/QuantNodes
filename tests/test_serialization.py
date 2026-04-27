# coding=utf-8
"""
序列化功能单元测试
"""

import pytest
import warnings
import os

from QuantNodes.core import Cond, Expression


class TestBasicSerialization:
    """基础序列化测试"""

    @pytest.fixture
    def simple_expr(self):
        """简单表达式"""
        return (Cond('a') > 5)._expr

    @pytest.fixture
    def complex_expr(self):
        """复杂表达式"""
        return (
            (Cond('close') > Cond('open')) &
            (Cond('volume') > 1000000) |
            (Cond.attr('metrics').sharpe >= 1.5)
        )._expr

    def test_json_roundtrip(self, simple_expr):
        """JSON 序列化往返"""
        json_str = simple_expr.to_json()
        restored = Expression.deserialize(json_str)
        assert str(restored) == str(simple_expr)

    def test_json_compact_roundtrip(self, simple_expr):
        """紧凑 JSON 序列化"""
        json_str = simple_expr.to_json(indent=None)
        restored = Expression.deserialize(json_str)
        assert str(restored) == str(simple_expr)

    def test_compact_roundtrip(self, complex_expr):
        """JSON+zlib 压缩序列化"""
        data = complex_expr.to_compact()
        assert isinstance(data, bytes)
        # 压缩应该比原始 JSON 小
        original_size = len(complex_expr.to_json(indent=None).encode())
        assert len(data) < original_size * 0.7  # 至少压缩 30%

        restored = Expression.deserialize(data)
        assert str(restored) == str(complex_expr)

    def test_json_bytes_roundtrip(self, simple_expr):
        """JSON bytes 序列化"""
        data = simple_expr.to_bytes()
        assert isinstance(data, bytes)
        restored = Expression.deserialize(data)
        assert str(restored) == str(simple_expr)


class TestMsgpackSerialization:
    """msgpack 序列化测试"""

    @pytest.fixture
    def expr(self):
        return (Cond('a') + Cond('b') > 10)._expr

    def test_msgpack_roundtrip(self, expr):
        """msgpack 序列化往返"""
        data = expr.to_msgpack()
        assert isinstance(data, bytes)
        restored = Expression.deserialize(data)
        assert str(restored) == str(expr)

    def test_msgpack_vs_json_size(self, expr):
        """msgpack 应该比 JSON 小"""
        msgpack_size = len(expr.to_msgpack())
        json_size = len(expr.to_json(indent=None).encode())
        # msgpack 通常更小（除非是非常简单的对象）
        assert msgpack_size < json_size or msgpack_size < json_size * 1.2


class TestProtobufSerialization:
    """Protobuf 序列化测试"""

    @pytest.fixture
    def expr(self):
        return (Cond('x') < Cond('y'))._expr

    def test_proto_roundtrip(self, expr):
        """Protobuf 序列化往返"""
        data = expr.to_proto()
        assert isinstance(data, bytes)
        restored = Expression.from_proto(data)
        assert str(restored) == str(expr)

    def test_int_preservation(self):
        """测试整数不被转成浮点数"""
        expr = (Cond('count') == 42)._expr
        restored = Expression.from_proto(expr.to_proto())
        # 常量表达式的值应该是整数
        assert '42' in str(restored)
        assert '42.0' not in str(restored)


class TestEncryptedSerialization:
    """加密序列化测试"""

    @pytest.fixture
    def expr(self):
        return (Cond('secret') == 'value')._expr

    def test_encrypted_with_password(self, expr):
        """密码模式加密"""
        password = "my-secret-key-123"
        encrypted = expr.to_encrypted(password)
        assert isinstance(encrypted, bytes)

        restored = Expression.from_encrypted(encrypted, password)
        assert str(restored) == str(expr)

    def test_encrypted_with_bytes_key(self, expr):
        """原始密钥模式加密"""
        key = os.urandom(32)  # AES-256 密钥
        encrypted = expr.to_encrypted(key)
        assert isinstance(encrypted, bytes)

        restored = Expression.from_encrypted(encrypted, key)
        assert str(restored) == str(expr)

    def test_wrong_password_fails(self, expr):
        """错误密码应该失败"""
        encrypted = expr.to_encrypted("correct-password")
        from cryptography.exceptions import InvalidTag

        with pytest.raises(InvalidTag):
            Expression.from_encrypted(encrypted, "wrong-password")

    def test_encrypted_non_deterministic(self, expr):
        """相同表达式每次加密结果应该不同（随机 nonce）"""
        password = "same-key"
        e1 = expr.to_encrypted(password)
        e2 = expr.to_encrypted(password)
        assert e1 != e2  # 由于随机 nonce，密文不同


class TestPickleSerialization:
    """Pickle 序列化测试（带警告）"""

    @pytest.fixture
    def expr(self):
        return (Cond('a') | Cond('b'))._expr

    def test_pickle_roundtrip(self, expr):
        """Pickle 序列化往返"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = expr.to_pickle()
            assert isinstance(data, bytes)
            restored = Expression.from_pickle(data)
        assert str(restored) == str(expr)

    def test_pickle_warning(self, expr):
        """Pickle 应该发出安全警告"""
        with pytest.warns(UserWarning, match="安全风险"):
            expr.to_pickle()


class TestAutoDetection:
    """自动格式检测测试"""

    def test_detect_json(self):
        """检测 JSON 字符串"""
        expr = (Cond('a') > 1)._expr
        json_str = expr.to_json()
        restored = Expression.deserialize(json_str)
        assert str(restored) == str(expr)

    def test_detect_json_bytes(self):
        """检测 JSON bytes"""
        expr = (Cond('a') > 1)._expr
        json_bytes = expr.to_bytes()
        restored = Expression.deserialize(json_bytes)
        assert str(restored) == str(expr)

    def test_detect_compact(self):
        """检测 zlib 压缩"""
        expr = (Cond('a') > 1)._expr
        compact = expr.to_compact()
        restored = Expression.deserialize(compact)
        assert str(restored) == str(expr)

    def test_detect_msgpack(self):
        """检测 msgpack"""
        expr = (Cond('a') > 1)._expr
        msg = expr.to_msgpack()
        restored = Expression.deserialize(msg)
        assert str(restored) == str(expr)


class TestSizeComparison:
    """不同方案大小对比测试"""

    def test_size_comparison(self):
        """验证压缩率符合预期"""
        expr = (
            (Cond('close') > Cond('open')) &
            (Cond('volume') > 1000000)
        )._expr

        sizes = {
            'json': len(expr.to_json(indent=None).encode()),
            'compact': len(expr.to_compact()),
            'msgpack': len(expr.to_msgpack()),
        }

        # JSON+zlib 应该最小
        assert sizes['compact'] < sizes['json']
        assert sizes['compact'] < sizes['msgpack'] * 0.9  # 至少比 msgpack 小 10%
        # 压缩率应该在 50% 以上
        assert sizes['compact'] < sizes['json'] * 0.5


class TestExpressionBuilderSerialization:
    """ExpressionBuilder 序列化支持测试"""

    def test_builder_serialize(self):
        """通过 ExpressionBuilder 直接序列化"""
        expr_builder = Cond('value') > 100
        # 需要访问 _expr 来序列化
        expr_obj = expr_builder._expr
        data = expr_obj.to_compact()
        restored = Expression.deserialize(data)
        assert 'value' in str(restored)
