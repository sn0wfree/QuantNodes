"""Tests for QuantNodes.research.prompts — registry/group/loader/renderer/store."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ── renderer ───────────────────────────────────────────────────


class TestRenderTemplate:
    """Tests for render_template (Jinja2 wrapper)."""

    def test_simple_variable(self):
        from QuantNodes.research.prompts.renderer import render_template

        assert render_template("Hello {{ name }}", name="World") == "Hello World"

    def test_multiple_variables(self):
        from QuantNodes.research.prompts.renderer import render_template

        out = render_template(
            "{{a}} + {{b}} = {{a + b}}", a=2, b=3
        )
        assert out == "2 + 3 = 5"

    def test_no_variables(self):
        from QuantNodes.research.prompts.renderer import render_template

        assert render_template("plain text") == "plain text"

    def test_unicode(self):
        from QuantNodes.research.prompts.renderer import render_template

        out = render_template("你好 {{ name }}", name="世界")
        assert out == "你好 世界"

    def test_conditional(self):
        from QuantNodes.research.prompts.renderer import render_template

        out = render_template(
            "{% if flag %}YES{% else %}NO{% endif %}", flag=True
        )
        assert out == "YES"


# ── group ──────────────────────────────────────────────────────


class TestPromptGroup:
    """Tests for PromptGroup dataclass."""

    def _make(self, **overrides):
        from QuantNodes.research.prompts.group import PromptGroup

        defaults = dict(
            name="test_prompt",
            version="1.0.0",
            source="unit",
            system="sys",
            user_template="Hello {{ name }}",
            feedback_template=None,
        )
        defaults.update(overrides)
        return PromptGroup(**defaults)

    def test_render_user(self):
        g = self._make()
        assert g.render_user(name="Alice") == "Hello Alice"

    def test_render_feedback_raises_when_none(self):
        g = self._make(feedback_template=None)
        with pytest.raises(ValueError, match="no feedback_template"):
            g.render_feedback(name="X")

    def test_render_feedback(self):
        g = self._make(feedback_template="Feedback: {{ msg }}")
        assert g.render_feedback(msg="good") == "Feedback: good"

    def test_is_compatible_same_major_higher_minor(self):
        g = self._make(version="1.2.3")
        assert g.is_compatible("1.2.0") is True
        assert g.is_compatible("1.5.0") is False
        assert g.is_compatible("2.0.0") is False
        assert g.is_compatible("0.9.0") is False  # major different

    def test_is_compatible_different_major(self):
        g = self._make(version="2.0.0")
        assert g.is_compatible("1.9.9") is False


# ── registry ───────────────────────────────────────────────────


class TestPromptRegistry:
    """Tests for PromptRegistry (multi-version prompt storage)."""

    def _group(self, name, version):
        from QuantNodes.research.prompts.group import PromptGroup

        return PromptGroup(
            name=name,
            version=version,
            source="unit",
            system="",
            user_template=f"tmpl for {name}@{version}",
            feedback_template=None,
        )

    def test_register_and_get_latest(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        reg.register(self._group("foo", "1.0.0"))
        reg.register(self._group("foo", "2.0.0"))
        reg.register(self._group("foo", "1.5.0"))

        g = reg.get("foo")  # latest by default
        assert g.version == "2.0.0"

    def test_get_specific_version(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        reg.register(self._group("foo", "1.0.0"))
        reg.register(self._group("foo", "2.0.0"))

        assert reg.get("foo", "1.0.0").version == "1.0.0"

    def test_get_missing_name_raises(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        with pytest.raises(KeyError, match="No prompt group named"):
            reg.get("missing")

    def test_get_missing_version_raises(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        reg.register(self._group("foo", "1.0.0"))
        with pytest.raises(KeyError, match="version '9.9.9'"):
            reg.get("foo", "9.9.9")

    def test_require_is_alias_for_get(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        reg.register(self._group("foo", "1.0.0"))
        assert reg.require("foo").name == "foo"
        assert reg.require("foo", "1.0.0").version == "1.0.0"

    def test_multiple_names_isolated(self):
        from QuantNodes.research.prompts.registry import PromptRegistry

        reg = PromptRegistry()
        reg.register(self._group("a", "1.0.0"))
        reg.register(self._group("b", "1.0.0"))
        assert reg.get("a").name == "a"
        assert reg.get("b").name == "b"


# ── loader ─────────────────────────────────────────────────────


class TestPromptLoader:
    """Tests for PromptLoader (YAML file → PromptGroup)."""

    def test_load_minimal_yaml(self, tmp_path: Path):
        from QuantNodes.research.prompts.loader import PromptLoader

        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "name": "minimal",
                    "version": "1.0.0",
                    "user_template": "Hello",
                }
            )
        )

        loader = PromptLoader(tmp_path)
        g = loader.load("minimal.yaml")
        assert g.name == "minimal"
        assert g.version == "1.0.0"
        assert g.user_template == "Hello"
        assert g.source == "custom"  # default
        assert g.feedback_template is None  # default

    def test_load_full_yaml(self, tmp_path: Path):
        from QuantNodes.research.prompts.loader import PromptLoader

        yaml_path = tmp_path / "full.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "name": "full",
                    "version": "2.1.0",
                    "source": "manual",
                    "system": "You are X",
                    "user_template": "Do {{ task }}",
                    "feedback_template": "Try again: {{ err }}",
                    "metadata": {"author": "alice"},
                }
            )
        )

        loader = PromptLoader(tmp_path)
        g = loader.load("full.yaml")
        assert g.name == "full"
        assert g.version == "2.1.0"
        assert g.source == "manual"
        assert g.system == "You are X"
        assert g.metadata == {"author": "alice"}
        assert g.render_user(task="Y") == "Do Y"
        assert g.render_feedback(err="boom") == "Try again: boom"

    def test_load_missing_file_raises(self, tmp_path: Path):
        from QuantNodes.research.prompts.loader import PromptLoader

        loader = PromptLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load("nope.yaml")


# ── store ──────────────────────────────────────────────────────


class TestPromptStore:
    """Tests for PromptStore (builtin + custom load)."""

    def test_empty_when_no_builtin_dir(self, tmp_path: Path, monkeypatch):
        from QuantNodes.research.prompts.store import PromptStore

        # builtin_dir 不存在应返回空 registry
        store = PromptStore(custom_dir=tmp_path)
        reg = store.load_builtin()
        assert len(reg._groups) == 0

    def test_load_builtin_from_directory(self, tmp_path: Path, monkeypatch):
        """通过 monkeypatch builtin_dir 来测试 load_builtin."""
        from QuantNodes.research.prompts.store import PromptStore

        # 创建临时 builtin 目录
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        (builtin / "a.yaml").write_text(
            yaml.safe_dump({"name": "a", "version": "1.0.0", "user_template": "AAA"})
        )
        (builtin / "b.yaml").write_text(
            yaml.safe_dump({"name": "b", "version": "2.0.0", "user_template": "BBB"})
        )

        store = PromptStore()
        monkeypatch.setattr(store, "builtin_dir", builtin)
        reg = store.load_builtin()

        assert reg.get("a").version == "1.0.0"
        assert reg.get("b").version == "2.0.0"

    def test_custom_dir_path_recorded(self, tmp_path: Path):
        from QuantNodes.research.prompts.store import PromptStore

        store = PromptStore(custom_dir=tmp_path)
        assert store.custom_dir == tmp_path

    def test_no_custom_dir_defaults_none(self):
        from QuantNodes.research.prompts.store import PromptStore

        store = PromptStore()
        assert store.custom_dir is None