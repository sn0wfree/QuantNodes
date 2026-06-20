# coding=utf-8
"""
CLI Enhanced 测试
"""



class TestChatCommand:
    def test_chat_help(self):
        """测试 chat --help 输出"""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "QuantNodes", "chat", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/ll/Public/QuantNodes",
        )
        assert result.returncode == 0
        assert "Agent" in result.stdout or "chat" in result.stdout.lower()

    def test_cmd_chat_exists(self):
        """测试 cmd_chat 函数存在"""
        from QuantNodes.cli import cmd_chat
        assert callable(cmd_chat)

    def test_enhanced_module_imports(self):
        """测试 enhanced 模块可以导入"""
        from QuantNodes.cli.enhanced import chat, chat_single
        assert callable(chat)
        assert callable(chat_single)
