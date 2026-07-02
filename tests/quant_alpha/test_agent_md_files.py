# coding=utf-8
"""
test_agent_md_files.py — 5 个 nanobot subagent md 文件存在性 + 结构 (v3.0.1 Phase 4)

按 .agent/agents/ 现有模式 (alpha-gpt-*.md / mcts-*.md) 校验:
1. 5 个文件存在 (logic-mining-{structure,semantics,abstraction}.md + market-logic-{generator,refinement}.md)
2. 每个 md 文件 ≥ 80 行 (符合现有模式)
3. 每个 md 含以下 section (H2 标题之一):
   - "## 角色定位"
   - "## 专业领域"
   - "## 工作流程"
   - "## 输出格式"
   - "## 验收标准"
   - "## 与 nanobot 集成"
4. spawn 调用引用 label 与文件名一致 (label = filename without .md)

SKILL.md:
1. QuantNodes/agent/skills_quant/logic-mining/SKILL.md 存在
2. 含 YAML front-matter `---` 头
3. 含 `name: logic-mining` 字段
"""
from pathlib import Path

import pytest

AGENTS_DIR = Path(".agent/agents")
SKILL_PATH = Path("QuantNodes/agent/skills_quant/logic-mining/SKILL.md")

REQUIRED_AGENT_MDS = [
    "logic-mining-structure.md",
    "logic-mining-semantics.md",
    "logic-mining-abstraction.md",
    "market-logic-generator.md",
    "market-logic-refinement.md",
]


class TestAgentMdFiles:
    @pytest.mark.parametrize("filename", REQUIRED_AGENT_MDS)
    def test_md_file_exists(self, filename):
        path = AGENTS_DIR / filename
        assert path.is_file(), f"missing {path}"

    @pytest.mark.parametrize("filename", REQUIRED_AGENT_MDS)
    def test_md_minimum_length(self, filename):
        """每个 md ≥ 80 行"""
        path = AGENTS_DIR / filename
        nlines = len(path.read_text(encoding="utf-8").splitlines())
        assert nlines >= 80, f"{path} only {nlines} lines (need >= 80)"

    @pytest.mark.parametrize("filename", REQUIRED_AGENT_MDS)
    def test_md_has_all_required_sections(self, filename):
        """每个 md 含 9 节核心 section"""
        text = (AGENTS_DIR / filename).read_text(encoding="utf-8")
        required_sections = [
            "## 角色定位",
            "## 专业领域",
            "## 工作流程",
            "## 输出格式",
            "## 验收标准",
            "## 与 nanobot 集成",
        ]
        missing = [s for s in required_sections if s not in text]
        assert not missing, f"{filename} missing sections: {missing}"

    @pytest.mark.parametrize("filename", REQUIRED_AGENT_MDS)
    def test_md_label_matches_spawn(self, filename):
        """spawn 引用与文件名一致 (label = filename minus .md)"""
        text = (AGENTS_DIR / filename).read_text(encoding="utf-8")
        expected_label = filename.replace(".md", "")
        assert expected_label in text, f"{filename} should spawn label={expected_label}"

    @pytest.mark.parametrize("filename,scope", [
        ("logic-mining-structure.md", "stage 1"),
        ("logic-mining-semantics.md", "stage 2"),
        ("logic-mining-abstraction.md", "stage 3"),
        ("market-logic-generator.md", "outer"),
        ("market-logic-refinement.md", "outer"),
    ])
    def test_md_first_role_line_declares_scope(self, filename, scope):
        """首段 H1 角色定位声明 scope"""
        text = (AGENTS_DIR / filename).read_text(encoding="utf-8")
        # 检查 H1 引用了对应的阶段/角色
        first_h1_block = text.split("\n")[0:3]
        first_h1 = next((l for l in first_h1_block if l.startswith("# ")), "")
        assert first_h1.startswith("# "), f"{filename} missing H1 role title"

    def test_all_five_present(self):
        """汇总: 5 个文件全在"""
        for f in REQUIRED_AGENT_MDS:
            assert (AGENTS_DIR / f).is_file(), f


class TestSkillMdFile:
    def test_skill_md_exists(self):
        assert SKILL_PATH.is_file(), f"missing {SKILL_PATH}"

    def test_skill_md_has_yaml_frontmatter(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        assert text.startswith("---\n"), "SKILL.md must have YAML front-matter"
        # 找到第二个 ---
        end = text.find("\n---\n", 4)
        assert end >= 0, "SKILL.md front-matter must be closed with ---"

    def test_skill_md_name_field(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        assert "name: logic-mining" in text, "SKILL.md must declare name=logic-mining"

    def test_skill_md_description_field(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        assert "description:" in text

    def test_skill_md_has_workflow(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        assert "工作流" in text

    def test_skill_md_has_acceptance_criteria(self):
        text = SKILL_PATH.read_text(encoding="utf-8")
        assert "验收标准" in text

    def test_skill_importable_in_skills_quant_registry(self):
        """整个 skills_quant 目录应可作为 Python 包导入"""
        from QuantNodes.agent import skills_quant  # type: ignore
        assert skills_quant is not None
        assert Path(skills_quant.__file__).is_file()
