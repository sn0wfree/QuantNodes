# coding=utf-8
"""
QuantNodes - AI-native quantitative research framework
"""

import subprocess
import sys
from pathlib import Path

__version__ = '0.4.1'
__author__ = 'sn0wfree'

_frontend_installed = False


def _check_and_install_frontend():
    """Install frontend dependencies if needed (only runs once)."""
    global _frontend_installed
    if _frontend_installed:
        return
    
    package_root = Path(__file__).parent.parent
    frontend_dir = package_root / "frontend"
    node_modules = frontend_dir / "node_modules"
    
    if frontend_dir.exists() and not node_modules.exists():
        print("检测到前端依赖未安装，正在安装...")
        try:
            subprocess.run(
                [sys.executable, "-m", "npm", "install"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True
            )
            print("✓ 前端依赖安装完成")
        except Exception as e:
            print(f"⚠ 前端依赖安装失败: {e}")
    
    _frontend_installed = True


_check_and_install_frontend()
