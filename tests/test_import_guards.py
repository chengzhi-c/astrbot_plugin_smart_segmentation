"""Guard the zero runtime dependencies invariant."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_astrbot_version_upper_bound_locked() -> None:
    """A4 删除了 on_decorating_result 的流式分支，前提是 DecorateStage 对
    STREAMING_RESULT 提前返回（astrbot 4.16-4.x 实测）。

    若上界抬到 5 及以上，必须先重新审视 A4：
    见 test_streaming_integration.py::test_decorate_stage_early_returns_for_streaming

    这条守卫只解析 metadata.yaml，不需要 astrbot：放在被 importorskip 的文件里
    会让"没装 astrbot 时改元数据"这个最可能的破坏场景恰好不运行。
    """
    meta = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    constraint = next(
        (
            line.split(":", 1)[1].strip().strip('"')
            for line in meta.splitlines()
            if line.startswith("astrbot_version:")
        ),
        "",
    )
    assert constraint, "metadata.yaml 缺少 astrbot_version 约束"
    assert "<5" in constraint, (
        f"astrbot_version 上界变更为 {constraint}，A4 的删除依据需重新验证"
    )


def test_no_third_party_imports() -> None:
    """All plugin .py files must only import stdlib or astrbot."""
    plugin_files = [
        f
        for f in ROOT.glob("**/*.py")
        if "tests" not in f.parts and "__pycache__" not in f.parts
    ]

    allowed_roots = sys.stdlib_module_names | {"astrbot"}

    violations: list[str] = []
    for file in plugin_files:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowed_roots:
                        violations.append(f"{file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root not in allowed_roots:
                    violations.append(f"{file.name}: from {node.module} import ...")

    assert not violations, "Third-party imports found:\n" + "\n".join(violations)
