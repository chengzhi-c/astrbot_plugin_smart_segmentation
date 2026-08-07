# CHANGELOG

## v0.2.2 (2026-08-07)

### Bug Fixes
- 修复运行时切换 `streaming_compat_enabled` 在纯流式流量下不生效的缺陷（D-1）。
  根因：流式结果在 `DecorateStage.process` 与 `RespondStage.process` 均提前返回，
  两个 hook 都不触发，补丁装卸失去驱动。改为 30 秒周期同步（`_sync_loop`）。
- 修复测试对全局 `asyncio.sleep` 的 monkeypatch 污染（D-1 暴露）：改 patch
  `calculate_send_delay`，避免影响常驻协程的时间语义。

### Refactoring
- 折叠 `streaming/patches.py` 中单目标补丁的未使用泛化（PatchHandle 单实例
  脚手架，净减 32 行；7 步状态机差分验证等价）
- 流式弱边界取窗口内最晚位置（O-13 B：段落更长、碎片更少）

### Testing
- 新增真实 Stage 流式集成测试（O-11）：把 O-1/A4 的一次性探针固化为常驻
  守卫（提前返回属实 / 补丁包装 / 卸载原样）
- 新增 `astrbot_version` 上界断言（O-12）：锁定 A4 的成立前提 <5
- 新增 D-1 周期同步测试（不改配置不发消息也能装卸补丁 / terminate 回收任务）
- 手工验证清单扩充至 14 项含部署步骤（真实 QQ 环境执行，见 docs/manual-verify.md）

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
