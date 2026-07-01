# coding=utf-8
"""QuantNodes quant-specific skills.

Each subdirectory contains a SKILL.md following the HKUDS/nanobot
skill convention (YAML front-matter + instruction body). These
skills are auto-discovered by nanobot at startup via the
`nanobot/skills/` integration; this __init__ merely makes the
directory a proper Python package so that ``setuptools.find_packages``
includes the SKILL.md files in the built sdist / wheel.

See:
- docs/14-上游nanobot升级指南.md
- QuantNodes/agent/config_mapper.py (skills section)
"""