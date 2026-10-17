from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PKG_NAME = "astrbot_plugin_smart_segmentation"

_ROOT = Path(__file__).resolve().parents[1]

# GitHub zip 解出来的目录名是 "<name>-main"，与包名不一致：注册一个别名，
# 让包内的相对导入能解析。目录名本就一致时无需别名，正常导入即可。
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

if _ROOT.name != PKG_NAME and PKG_NAME not in sys.modules:
    init_file = _ROOT / "__init__.py"
    if not init_file.exists():
        raise RuntimeError(f"{PKG_NAME} 缺少 __init__.py：{init_file}")
    spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        init_file,
        submodule_search_locations=[str(_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法构造 {PKG_NAME} 的模块 spec：{init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG_NAME] = module
    spec.loader.exec_module(module)
