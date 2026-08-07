# CHANGELOG

## v0.2.0

- 拆分模块：settings / segmenter / follow_up / chain_utils / streaming / errors
- 配置上下限单一来源（`bounds.py` 对齐 `_conf_schema.json`）；streaming min/max 互斥校正
- `SettingsLoader` 按配置值指纹缓存，hook 热路径不再全量重解析
- 移除旧版兼容：`get_using_provider` 回退、位置参数探测（锁定 AstrBot `>=4.16,<5`）
- 流式 C-light：默认不 import/不装补丁；仅 `ResultDecorateStage`；sync 幂等短路
- 修复流式混合链组件顺序（At/Plain 保持原始顺序）；删死代码与误导注释
- follow-up 簿记（去重/pending/清态）收进 `FollowUpDispatcher`
- pytest 79 项：纯逻辑、settings 缓存、boundary、segmenter（超时/取消/无 provider）、
  follow_up（失败停止/guard）、chain_utils、stream wrapper、patch 多场景、
  端到端流程、金样
