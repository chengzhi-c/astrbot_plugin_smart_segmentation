# CHANGELOG

## v0.2.1 (2026-08-07)

### Testing & Quality
- 新增 schema 默认值对齐守卫（`_conf_schema.json` default ↔ dataclass default）
- 新增零运行时依赖 import 边界守卫（相对导入按 AST `level` 判定）
- 金样扩充至 17+13（从 9+9）：未闭合括号、嵌套括号、纯标点、CRLF、
  未闭合 thinking、emoji 边界、长文本 cap、尾随垃圾、无标签 fence 含代码、
  数组内换行、英文括号等
- 单元测试补充：`extract_json_array_text` 无括号、`cap_segments` max=1、
  嵌套括号合并、流式边界（换行在 0 位/弱边界不足 min/无边界硬切）

### Refactoring
- `_CONFIG_KEYS` 从 dataclass 字段自动派生（消除 14 行手工维护）
- 删除 `_stage_lines` 中不可达兜底（差分 4000 例实证等价）
- Jitter 常量收敛至 `bounds.py`（`DELAY_JITTER_SECONDS`）
- shutdown 取消日志降为 DEBUG（高频预期路径降噪）
- 删除 `on_decorating_result` 中不可达流式分支（`ResultDecorateStage` 对
  `STREAMING_RESULT` 提前返回；流式兼容仅由补丁路径承担，`metadata.yaml`
  锁定 `<5`）

### Chore
- 删除无引用 `data/` 目录（cmd_config.json + t2i_templates/，-1313 行）

### Documentation
- requirements-dev.txt 补充 pytest-asyncio 与 astrbot 依赖
- 新增 docs/manual-verify.md 发布前手工验证清单

## v0.2.0

- 拆分模块：settings / segmenter / follow_up / chain_utils / streaming / errors
- 配置上下限单一来源（`bounds.py` 对齐 `_conf_schema.json`）；streaming min/max 互斥校正
- `SettingsLoader` 按配置值指纹缓存，hook 热路径不再全量重解析
- 移除旧版兼容：`get_using_provider` 回退、位置参数探测（锁定 AstrBot `>=4.16,<5`）
- 流式 C-light：默认不 import/不装补丁；仅 `ResultDecorateStage`；sync 幂等短路
- 修复流式混合链组件顺序（At/Plain 保持原始顺序）；删死代码与误导注释
- follow-up 簿记（去重/pending/清态）收进 `FollowUpDispatcher`
- pytest 95 项：纯逻辑、settings 缓存、boundary、segmenter（超时/取消/无 provider）、
  follow_up（失败停止/guard）、chain_utils、stream wrapper、patch 多场景、
  端到端流程、金样
