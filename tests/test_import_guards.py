"""Guard the zero runtime dependencies invariant."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
